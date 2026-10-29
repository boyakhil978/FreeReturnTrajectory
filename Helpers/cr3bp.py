'''Planar circular restricted three-body problem: dynamics, initial conditions,
derived quantities, and the free-return marcher with event location.

All quantities are nondimensional (L, T, V from constants.py) unless a name
says otherwise. Every function takes mu explicitly so that the verification
tests (mu = 0 Kepler, JPL catalog mu) run through exactly the same code as the
results.
'''

import numpy as np
from scipy.optimize import brentq

from .constants import MU, V, R_0_ND, R_EI_ND, R_MOON_ND, L
from .rk4 import rk4_step


# ---------------------------------------------------------------- dynamics

def f(u, t=0.0, mu=MU):
    '''Right-hand side u_dot = f(u, t) of the planar CR3BP in the rotating frame.

    u = [x, y, v_x, v_y]. Earth sits at (-mu, 0), the Moon at (1 - mu, 0).
    t is unused (the system is autonomous) but kept so every marcher can call
    f(u, t) the same way.
    '''
    r1 = np.sqrt((u[0] + mu)**2 + u[1]**2)
    r2 = np.sqrt((u[0] - 1 + mu)**2 + u[1]**2)

    return np.array([
        u[2],
        u[3],
        (1 - (1 - mu)/r1**3 - mu/r2**3)*u[0] + 2*u[3] + mu*(1 - mu)*(1/r2**3 - 1/r1**3),
        (1 - (1 - mu)/r1**3 - mu/r2**3)*u[1] - 2*u[2]
    ])


def make_f(mu):
    '''f with mu fixed, in the f(u, t) form that rk4 expects.'''
    return lambda u, t: f(u, t, mu)


# ---------------------------------------------------------------- initial condition

def tli_state(delta_v, phi, r0=R_0_ND, mu=MU):
    '''Rotating-frame state right after a tangential TLI burn.

    delta_v : TLI burn magnitude in km/s (dimensional)
    phi : position angle on the circular parking orbit, rad, from the Earth-Moon line
    r0 : parking orbit radius (nondimensional)
    '''
    x_0 = -mu + r0*np.cos(phi)
    y_0 = r0*np.sin(phi)

    v_c = np.sqrt((1 - mu)/r0) + delta_v/V

    # inertial velocity, then v_rot = v_in - omega x r (includes Earth's motion about the barycenter)
    vx_0 = -v_c*np.sin(phi) + y_0
    vy_0 = v_c*np.cos(phi) - (x_0 + mu)

    return np.array([x_0, y_0, vx_0, vy_0])


def parking_orbit_rate(r0=R_0_ND, mu=MU):
    '''Angular rate n_LEO of the parking orbit; a timing error dt shifts phi by n_LEO*dt.'''
    return np.sqrt((1 - mu)/r0**3)


# ---------------------------------------------------------------- derived quantities

def earth_relative(u, mu=MU):
    '''Earth-relative position r_E and Earth-relative inertial velocity v_E.'''
    r_E = np.array([u[0] + mu, u[1]])
    v_E = np.array([u[2] - u[1], u[3] + u[0] + mu])
    return r_E, v_E


def flight_path_angle(u, mu=MU):
    '''Inertial flight-path angle relative to Earth, in degrees (negative = descending).'''
    r_E, v_E = earth_relative(u, mu)
    return np.degrees(np.arcsin(r_E @ v_E / (np.linalg.norm(r_E)*np.linalg.norm(v_E))))


def jacobi(u, mu=MU):
    '''Jacobi constant C = 2*Omega - (v_x^2 + v_y^2); works on one state or an array of states.'''
    u = np.asarray(u)
    x, y, vx, vy = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)
    return x**2 + y**2 + 2*(1 - mu)/r1 + 2*mu/r2 - vx**2 - vy**2


def apply_burn(u, dv, mu=MU):
    '''Impulsive burn of dv km/s along the Earth-relative inertial velocity.

    A burn changes the inertial velocity; since v_rot = v_in - omega x r and the
    position does not change, the same vector is added to the rotating-frame velocity.
    '''
    _, v_E = earth_relative(u, mu)
    u_new = u.copy()
    u_new[2:] += (dv/V) * v_E/np.linalg.norm(v_E)
    return u_new


# ---------------------------------------------------------------- free-return marcher

def fly_to_entry(u0, dt, t_max=3.0, mu=MU, t_c=None, dv_c=0.0, flyby_radius=0.1, store=False):
    '''March a free-return trajectory with fixed-step RK4 until entry interface.

    Entry interface is the first crossing of |r_E| = R_EI after the spacecraft
    has come within flyby_radius of the Moon. The crossing is located with a
    partial RK4 step of length tau in (0, dt), found by root-finding, so the
    event inherits RK4's fourth-order accuracy (interpolating between steps
    would not).

    t_c, dv_c : optional midcourse correction of dv_c km/s along v_E, applied
                at the step nearest t_c (the step k with k*dt closest to t_c).

    Returns a dict with
        status : 'entry', 'lunar impact', or 'no return'
        t_ei, u_ei, gamma_ei, v_ei_kms : entry time, state, flight-path angle (deg), inertial speed (km/s)
        perilune_alt_km, t_perilune : closest sampled approach to the Moon
        u, times : full history (only if store=True)
    '''
    func = make_f(mu)
    n_max = int(round(t_max/dt))
    k_c = None if t_c is None else int(round(t_c/dt))

    u = np.array(u0, dtype=float)
    passed_moon = False
    r2_min, t_r2_min = np.inf, 0.0
    hist = [u.copy()] if store else None

    def r_earth(state):
        return np.hypot(state[0] + mu, state[1])

    out = {'status': 'no return'}

    for k in range(n_max):
        t = k*dt

        if k_c is not None and k == k_c:
            u = apply_burn(u, dv_c, mu)

        u_next = rk4_step(func, u, t, dt)

        r2 = np.hypot(u_next[0] - 1 + mu, u_next[1])
        if r2 < r2_min:
            r2_min, t_r2_min = r2, t + dt
        if r2 < R_MOON_ND:
            out['status'] = 'lunar impact'
            break
        if r2 < flyby_radius:
            passed_moon = True

        if passed_moon and r_earth(u) > R_EI_ND >= r_earth(u_next):
            g = lambda tau: r_earth(rk4_step(func, u, t, tau)) - R_EI_ND
            tau = brentq(g, 0.0, dt, xtol=1e-15, rtol=4*np.finfo(float).eps, maxiter=200)
            u_ei = rk4_step(func, u, t, tau)
            _, v_E = earth_relative(u_ei, mu)
            out.update(status='entry', t_ei=t + tau, u_ei=u_ei,
                       gamma_ei=flight_path_angle(u_ei, mu), v_ei_kms=np.linalg.norm(v_E)*V)
            if store:
                hist.append(u_ei.copy())
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


def entry_angle(delta_v, phi, dt, **kw):
    '''gamma_EI (deg) for a TLI of delta_v km/s at phi; nan if the run does not reach entry.'''
    res = fly_to_entry(tli_state(delta_v, phi), dt, **kw)
    return res['gamma_ei'] if res['status'] == 'entry' else np.nan
