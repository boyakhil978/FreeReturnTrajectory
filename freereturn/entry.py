'''Free-return marcher: TLI to entry interface with fixed-step RK4, without storing more than
needed, plus an optional impulsive midcourse correction. Used for targeting and for the
time-step and sensitivity studies, which need hundreds of runs.
'''

import numpy as np

from freereturn.constants import MU, L, R_EI_ND, R_MOON_ND
from freereturn.dynamics import f_spacecraft
from freereturn.integrators import rk4_step
from freereturn.trajectory import tli_state, apply_burn, locate_entry, inertial_speed_fpa

def fly_to_entry(u0, dt, t_max=3.0, t_c=None, dv_c=0.0, flyby_radius=0.1, store=False):
    '''March from u0 until the first entry-interface crossing after the lunar flyby.

    dt, t_max : nondimensional time step and give-up time (3 time units is about 13 days)
    t_c, dv_c : optional midcourse correction of dv_c [km/s] along the Earth-centred inertial
                velocity, applied at the step whose time is nearest t_c
    flyby_radius : the spacecraft counts as having flown past the Moon once it is this close
                   (nondimensional; 0.1 is about 38,000 km)

    Returns a dict with
        status : 'entry', 'lunar impact' or 'no return'
        t_ei, u_ei, gamma_ei, v_ei_kms : entry time, state, flight-path angle [deg], speed [km/s]
        perilune_alt_km, t_perilune : closest sampled approach to the Moon
        u, times : full history, only if store=True
    '''
    f = lambda u, t: f_spacecraft(u, t, MU)
    n_max = int(round(t_max/dt))
    k_c = None if t_c is None else int(round(t_c/dt))

    u = np.array(u0, dtype=float)
    passed_moon = False
    r2_min, t_r2_min = np.inf, 0.0
    hist = [u.copy()] if store else None
    out = {'status': 'no return'}

    for k in range(n_max):
        t = k*dt
        if k == k_c:
            u = apply_burn(u, dv_c)

        u_next = rk4_step(f, u, t, dt)

        r2 = np.hypot(u_next[0] - 1 + MU, u_next[1])
        if r2 < r2_min:
            r2_min, t_r2_min = r2, t + dt
        if r2 < R_MOON_ND:
            out['status'] = 'lunar impact'
            break
        passed_moon = passed_moon or r2 < flyby_radius

        r1, r1_next = np.hypot(u[0] + MU, u[1]), np.hypot(u_next[0] + MU, u_next[1])
        if passed_moon and r1 > R_EI_ND >= r1_next:
            u_ei, tau = locate_entry(u, t, dt)
            v_ei, gamma = inertial_speed_fpa(u_ei)
            out.update(status='entry', t_ei=t + tau, u_ei=u_ei, gamma_ei=gamma, v_ei_kms=v_ei)
            if store:
                hist.append(u_ei)
            break

        u = u_next
        if store:
            hist.append(u.copy())

    out['perilune_alt_km'] = (r2_min - R_MOON_ND)*L
    out['t_perilune'] = t_r2_min
    if store:
        out['u'] = np.array(hist)
        out['times'] = np.arange(len(hist))*dt
        if out['status'] == 'entry':
            out['times'][-1] = out['t_ei']
    return out

def entry_angle(delta_v, phi, dt, **kwargs):
    '''Entry flight-path angle [deg] for a tangential TLI of delta_v [km/s] at phi [rad];
    nan if the run does not reach entry interface.'''
    res = fly_to_entry(tli_state(delta_v, phi), dt, **kwargs)
    return res['gamma_ei'] if res['status'] == 'entry' else np.nan
