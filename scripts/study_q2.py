'''Q2: cost of a single impulsive midcourse correction that restores gamma_EI = -6.52 deg after a
TLI magnitude error, as a function of the correction time t_c.

For each injection error, the uncorrected trajectory is marched once and stored. For each t_c the
run restarts from the stored state at t_c, applies an impulsive dv_c and marches to entry interface.
Two burn directions are solved for:
  along  - along the Earth-centred inertial velocity v_E (negative = retrograde), as planned;
  normal - in-plane, perpendicular to v_E (positive = 90 deg counter-clockwise from v_E).
dv_c is the root of gamma_EI(dv_c) = -6.52, found by expanding a bracket outward from zero in both
directions and refining with Brent's method; the smaller-magnitude root is kept. Runs that never reach entry interface are counted as gamma = 0
(the shallow side), as in retarget_nominal.py.

Writes results/q2.json and figures/q2_correction_cost.pdf.
Run from the repository root (after retarget_nominal.py):  python scripts/study_q2.py
'''

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent)) # scripts/, for plot_style

from freereturn.constants import MU, T, V, GAMMA_TARGET
from freereturn.trajectory import tli_state, inertial_speed_fpa
from freereturn.entry import fly_to_entry
from plot_style import plt, save, BLUE, RED, GREY

INJECTION_ERRORS = [1.0, -1.0] # m/s
T_C_DAYS = np.linspace(0.1, 6.5, 18) # from just after TLI to about one day before entry
BRACKET_STEPS = 1e-3*np.array([1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]) # km/s
DIRECTIONS = {'along': 0.0, 'normal': np.pi/2} # angle from v_E
FLYBY_RADIUS = 0.1 # same definition as fly_to_entry


def nominal():
    return json.loads((ROOT/'results'/'nominal.json').read_text())


def stored_trajectory(error_ms):
    nom = nominal()
    return fly_to_entry(tli_state(nom['delta_v_kms'] + error_ms*1e-3, nom['phi_rad']), nom['dt_nd'], store=True)


def burn(u, dv, angle):
    """Impulsive burn of dv [km/s] at angle [rad] counter-clockwise from the Earth-centred inertial velocity."""
    v_in = np.array([u[2] - u[1], u[3] + u[0] + MU])
    c, s = np.cos(angle), np.sin(angle)
    direction = np.array([c*v_in[0] - s*v_in[1], s*v_in[0] + c*v_in[1]])/np.linalg.norm(v_in)
    u_new = np.array(u, dtype=float)
    u_new[2:] += dv/V*direction
    return u_new


def solve_correction(args):
    error_ms, t_c_days, direction = args
    dt = nominal()['dt_nd']
    run = stored_trajectory(error_ms)
    t_c = t_c_days*86400/T

    k = int(round(t_c/dt))
    u_c, t_k = run['u'][k], run['times'][k]
    passed = bool(np.min(np.hypot(run['u'][:k + 1, 0] - 1 + MU, run['u'][:k + 1, 1])) < FLYBY_RADIUS)
    angle = DIRECTIONS[direction]

    def fly(dv_c):
        return fly_to_entry(burn(u_c, dv_c, angle), dt, t0=t_k, passed_moon=passed)

    def miss(dv_c):
        res = fly(dv_c)
        return (res['gamma_ei'] if res['status'] == 'entry' else 0.0) - GAMMA_TARGET

    row = {'error_ms': error_ms, 't_c_days': t_c_days, 'direction': direction, 'after_flyby': passed,
           'speed_kms': inertial_speed_fpa(u_c)[0], 'dv_c_ms': None, 'gamma_ei': None}

    g0 = miss(0.0)
    prev = {+1: (0.0, g0), -1: (0.0, g0)}
    for step in BRACKET_STEPS:
        roots = []
        for sign in (+1, -1):
            g = miss(sign*step)
            a, g_a = prev[sign]
            if np.sign(g) != np.sign(g_a):
                roots.append(brentq(miss, a, sign*step, xtol=1e-9))
            prev[sign] = (sign*step, g)
        if roots:
            dv_c = min(roots, key=abs)
            row.update(dv_c_ms=dv_c*1e3, gamma_ei=fly(dv_c).get('gamma_ei'))
            break
    return row


def main():
    nom = nominal()
    tasks = [(e, t, d) for e in INJECTION_ERRORS for t in T_C_DAYS for d in DIRECTIONS]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(solve_correction, tasks))

    uncorrected = {}
    for e in INJECTION_ERRORS:
        res = fly_to_entry(tli_state(nom['delta_v_kms'] + e*1e-3, nom['phi_rad']), nom['dt_nd'])
        uncorrected[e] = res['gamma_ei'] if res['status'] == 'entry' else res['status']

    (ROOT/'results'/'q2.json').write_text(json.dumps({'uncorrected_gamma': {str(k): v for k, v in uncorrected.items()},
                                                       't_perilune_days': nom['t_perilune_days'],
                                                       't_ei_days': nom['t_ei_days'],
                                                       'corrections': rows}, indent=2))

    for e, g in uncorrected.items():
        print(f'uncorrected, {e:+.1f} m/s: {g}')
    for r in rows:
        dv = f'{r["dv_c_ms"]:+9.4f} m/s' if r['dv_c_ms'] is not None else '  not found'
        print(f'error {r["error_ms"]:+.1f} m/s   t_c = {r["t_c_days"]:5.2f} d   {r["direction"]:<6}  dv_c = {dv}   '
              f'|v_E| = {r["speed_kms"]:.3f} km/s   gamma = {r["gamma_ei"]}   after flyby: {r["after_flyby"]}')

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axvline(nom['t_perilune_days'], color=GREY, lw=6, alpha=0.25)
    ax.text(nom['t_perilune_days'], 0.97, ' lunar flyby', transform=ax.get_xaxis_transform(), va='top', fontsize=8, color='#333333')
    ax.axhline(1.0, color=GREY, ls=':', lw=1, label='size of the injection error (1 m/s)')

    for e, colour in zip(INJECTION_ERRORS, (BLUE, RED)):
        for direction, ls, marker in (('along', '-', 'o'), ('normal', '--', 's')):
            rs = [r for r in rows if r['error_ms'] == e and r['direction'] == direction and r['dv_c_ms'] is not None]
            t = np.array([r['t_c_days'] for r in rs])
            dv = np.abs([r['dv_c_ms'] for r in rs])
            ax.semilogy(t, dv, ls=ls, marker=marker, ms=4, color=colour, lw=1.3,
                        mfc=colour if direction == 'along' else 'white',
                        label=fr'TLI {e:+.0f} m/s, burn {"along" if direction == "along" else "normal to"} $\mathbf{{v}}_E$')

    ax.set_xlabel(r'correction time $t_c$ (days after TLI)')
    ax.set_ylabel(r'correction $|\Delta v_c|$ (m/s)')
    ax.legend(fontsize=7.5, loc='upper left')
    save(fig, 'q2_correction_cost')


if __name__ == '__main__':
    main()
