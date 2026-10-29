import numpy as np
from scipy.optimize import brentq

from freereturn.constants import MU, L, T, V, R_EARTH, R_MOON, R_EI, R_0
from freereturn.dynamics import f_spacecraft
from freereturn.integrators import rk4_step

# Every helper takes optional length / time / velocity units (lu [km], tu [s], vu [km/s]) so the
# same nondimensional model can be scaled to a different Earth-Moon distance. See unit_scales().

def unit_scales(lu):
    '''Time and velocity units for an Earth-Moon distance lu [km], from Kepler's third law (T ~ L^1.5).'''
    tu = T * (lu/L)**1.5
    return tu, lu/tu

def earth_dist(u, lu=L):
    return np.hypot(u[..., 0] + MU, u[..., 1]) * lu # km

def moon_dist(u, lu=L):
    return np.hypot(u[..., 0] - 1 + MU, u[..., 1]) * lu # km

def circular_speed(r):
    '''Speed of a circular orbit around Earth at radius r [km], in km/s.'''
    return np.sqrt((1 - MU)/(r/L)) * V

def inertial_speed_fpa(u, vu=V):
    '''Earth-centred inertial speed (km/s) and flight-path angle (deg) of a single state.

    With omega = 1, v_in = v_rot + omega x r_E = [v_x - y, v_y + (x + mu)].
    '''
    r = np.array([u[0] + MU, u[1]])
    v_in = np.array([u[2] - u[1], u[3] + (u[0] + MU)])
    fpa = np.degrees(np.arcsin(r @ v_in / (np.linalg.norm(r) * np.linalg.norm(v_in))))
    return np.linalg.norm(v_in) * vu, fpa

def initial_state(r0, speed, fpa, phi, lu=L, vu=V):
    '''Nondimensional rotating-frame state from an Earth-centred inertial burnout state.

    r0 [km], speed [km/s], fpa [deg] above local horizontal, phi [rad] position angle around Earth.
    '''
    r0_ND, v_ND, gam = r0/lu, speed/vu, np.radians(fpa)

    x_0 = -MU + r0_ND*np.cos(phi)
    y_0 = r0_ND*np.sin(phi)

    # radial + prograde (counter-clockwise) components
    vx_in = v_ND*(np.sin(gam)*np.cos(phi) - np.cos(gam)*np.sin(phi))
    vy_in = v_ND*(np.sin(gam)*np.sin(phi) + np.cos(gam)*np.cos(phi))

    return np.array([x_0, y_0, vx_in + y_0, vy_in - (x_0 + MU)])

def rotating_frame(r_moon, v_moon):
    '''Rows are the rotating-frame axes in inertial coordinates: x toward the Moon, z along the
    Moon's orbital angular momentum, y completing the right-handed set.'''
    x_hat = r_moon / np.linalg.norm(r_moon)
    h = np.cross(r_moon, v_moon)
    z_hat = h / np.linalg.norm(h)
    return np.array([x_hat, np.cross(z_hat, x_hat), z_hat])

def _rotation_between(a, b):
    '''Rotation matrix taking unit vector a onto unit vector b (Rodrigues' formula).'''
    k = np.cross(a, b)
    s, c = np.linalg.norm(k), a @ b
    if s < 1e-12:
        return np.eye(3)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]) / s
    return np.eye(3) + s*K + (1 - c)*K @ K

def state_from_inertial(r, v, r_moon, v_moon, lu=L, vu=V, method='rotate'):
    '''Nondimensional rotating-frame state from Earth-centred inertial r [km], v [km/s].

    method='rotate'  : tilt the spacecraft's own orbit plane into the Moon's orbital plane about
                       their line of nodes. Keeps |r|, |v| and the flight-path angle, so the
                       Earth-relative conic (energy, perigee) is unchanged.
    method='project' : drop the out-of-plane components. Fine when the trajectory is already
                       close to the Moon's plane, but changes |r|, |v| and the perigee when it is not.
    '''
    R = rotating_frame(r_moon, v_moon)

    if method == 'rotate':
        h = np.cross(r, v)
        align = _rotation_between(h/np.linalg.norm(h), R[2]*np.sign(h @ R[2]))
        r, v = align @ r, align @ v

    r_rot, v_in = R @ r, R @ v

    x_0 = -MU + r_rot[0]/lu
    y_0 = r_rot[1]/lu

    return np.array([x_0, y_0, v_in[0]/vu + y_0, v_in[1]/vu - (x_0 + MU)])

def trajectory_events(u, times_ND, lu=L, tu=T):
    '''Closest lunar approach and first entry-interface crossing after it.'''
    r_E, r_M = earth_dist(u, lu), moon_dist(u, lu)
    t_h = times_ND * tu / 3600

    k = np.argmin(r_M)
    events = {'perilune_alt': r_M[k] - R_MOON, 't_perilune': t_h[k], 'perilune_idx': k}

    crossings = np.where((r_E[k:-1] > R_EI) & (r_E[k + 1:] <= R_EI))[0]
    if len(crossings) == 0:
        j = k + np.argmin(r_E[k:])
        events.update(entered=False, return_perigee_alt=r_E[j] - R_EARTH, t_ei=t_h[j], ei_idx=j)
    else:
        j = k + crossings[0]
        u_ei, tau = locate_entry(u[j], times_ND[j], times_ND[j + 1] - times_ND[j], lu)
        events.update(entered=True, return_perigee_alt=None, t_ei=t_h[j] + tau*tu/3600, ei_idx=j)
        events['ei_speed'], events['ei_fpa'] = inertial_speed_fpa(u_ei, lu/tu)

    return events

def locate_entry(u_k, t_k, dt, lu=L, mu=MU):
    '''State at entry interface inside the step that crosses it.

    Finds tau in (0, dt) such that one RK4 step of length tau from u_k lands exactly on
    |r_E| = R_EI. This keeps the event fourth-order accurate; interpolating linearly between
    the two steps would only be second order.

    Returns
    -------
    u_ei : ndarray, the state at entry interface
    tau : float, time from t_k to entry interface (nondimensional)
    '''
    f = lambda u, t: f_spacecraft(u, t, mu)
    g = lambda tau: earth_dist(rk4_step(f, u_k, t_k, tau), lu) - R_EI
    tau = brentq(g, 0.0, dt, xtol=1e-15, rtol=4*np.finfo(float).eps, maxiter=200)
    return rk4_step(f, u_k, t_k, tau), tau

def tli_state(delta_v, phi, r0=R_0):
    '''Rotating-frame state right after a tangential TLI burn of delta_v [km/s] from a circular
    parking orbit of radius r0 [km], at position angle phi [rad] from the Earth-Moon line.'''
    return initial_state(r0, circular_speed(r0) + delta_v, 0.0, phi)

def apply_burn(u, dv, vu=V, mu=MU):
    '''Impulsive burn of dv [km/s] along the Earth-centred inertial velocity.

    The position does not change, so the inertial and rotating-frame velocities change by the
    same vector.
    '''
    v_in = np.array([u[2] - u[1], u[3] + u[0] + mu])
    u_new = np.array(u, dtype=float)
    u_new[2:] += dv/vu * v_in/np.linalg.norm(v_in)
    return u_new
