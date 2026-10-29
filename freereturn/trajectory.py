import numpy as np

from freereturn.constants import MU, L, T, V, R_EARTH, R_MOON, R_EI

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
        # linearly interpolate between the steps on either side of R_EI
        s = (r_E[j] - R_EI) / (r_E[j] - r_E[j + 1])
        u_ei = u[j] + s*(u[j + 1] - u[j])
        events.update(entered=True, return_perigee_alt=None, t_ei=t_h[j] + s*(t_h[j + 1] - t_h[j]), ei_idx=j)
        events['ei_speed'], events['ei_fpa'] = inertial_speed_fpa(u_ei, lu/tu)

    return events
