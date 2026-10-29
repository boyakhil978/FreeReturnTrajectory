import numpy as np

from freereturn.constants import MU, L, T, V, R_EARTH, R_MOON, R_EI

def earth_dist(u):
    return np.hypot(u[..., 0] + MU, u[..., 1]) * L # km

def moon_dist(u):
    return np.hypot(u[..., 0] - 1 + MU, u[..., 1]) * L # km

def circular_speed(r):
    '''Speed of a circular orbit around Earth at radius r [km], in km/s.'''
    return np.sqrt((1 - MU)/(r/L)) * V

def inertial_speed_fpa(u):
    '''Earth-centred inertial speed (km/s) and flight-path angle (deg) of a single state.

    With omega = 1, v_in = v_rot + omega x r_E = [v_x - y, v_y + (x + mu)].
    '''
    r = np.array([u[0] + MU, u[1]])
    v_in = np.array([u[2] - u[1], u[3] + (u[0] + MU)])
    fpa = np.degrees(np.arcsin(r @ v_in / (np.linalg.norm(r) * np.linalg.norm(v_in))))
    return np.linalg.norm(v_in) * V, fpa

def initial_state(r0, speed, fpa, phi):
    '''Nondimensional rotating-frame state from an Earth-centred inertial burnout state.

    r0 [km], speed [km/s], fpa [deg] above local horizontal, phi [rad] position angle around Earth.
    '''
    r0_ND, v_ND, gam = r0/L, speed/V, np.radians(fpa)

    x_0 = -MU + r0_ND*np.cos(phi)
    y_0 = r0_ND*np.sin(phi)

    # radial + prograde (counter-clockwise) components
    vx_in = v_ND*(np.sin(gam)*np.cos(phi) - np.cos(gam)*np.sin(phi))
    vy_in = v_ND*(np.sin(gam)*np.sin(phi) + np.cos(gam)*np.cos(phi))

    return np.array([x_0, y_0, vx_in + y_0, vy_in - (x_0 + MU)])

def trajectory_events(u, times_ND):
    '''Closest lunar approach and first entry-interface crossing after it.'''
    r_E, r_M = earth_dist(u), moon_dist(u)
    t_h = times_ND * T / 3600

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
        events['ei_speed'], events['ei_fpa'] = inertial_speed_fpa(u_ei)

    return events
