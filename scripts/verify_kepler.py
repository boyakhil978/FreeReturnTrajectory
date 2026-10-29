'''Verification 1: RK4 against the exact two-body solution (mu = 0).

With mu = 0 the CR3BP reduces to Kepler motion seen from a frame rotating at
unit rate. We integrate an exact ellipse with perigee at the parking-orbit
radius and e = 0.97 (close to the translunar ellipse) through the project's
own f and rk4, and compare with the exact solution from Kepler's equation.

Run from the repository root:  python scripts/verify_kepler.py
'''

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent)) # scripts/, for plot_style

from Helpers.constants import R_0_ND, L, T
from Helpers.cr3bp import make_f
from Helpers.rk4 import rk4
from plot_style import plt, save, BLUE, RED

E_ECC = 0.97
A = R_0_ND/(1 - E_ECC) # semi-major axis, from r_p = a (1 - e)
B = A*np.sqrt(1 - E_ECC**2) # semi-minor axis
N = A**-1.5 # mean motion (GM = 1)
P = 2*np.pi/N # period


def kepler_exact_rot(t):
    '''Exact mu = 0 state in the rotating frame, perigee at t = 0.'''
    M, E = N*t, np.pi # E_0 = pi is robust for e close to 1
    for _ in range(100): # Newton on M = E - e sin E
        dE = (E - E_ECC*np.sin(E) - M)/(1 - E_ECC*np.cos(E))
        E -= dE
        if abs(dE) < 1e-15:
            break
    den = 1 - E_ECC*np.cos(E)
    X, Y = A*(np.cos(E) - E_ECC), B*np.sin(E) # inertial position
    VX, VY = -A*N*np.sin(E)/den, B*N*np.cos(E)/den # inertial velocity
    c, s = np.cos(t), np.sin(t)
    x, y = c*X + s*Y, -s*X + c*Y # r_rot = R(-t) r_in
    return np.array([x, y, c*VX + s*VY + y, -s*VX + c*VY - x]) # v_rot = R(-t) v_in + (y, -x)


def main():
    f0 = make_f(0.0)
    u0 = kepler_exact_rot(0.0)
    print(f'a = {A:.4f}, P = {P:.4f} ({P*T/86400:.2f} days), v_p = {u0[3] + u0[0]:.4f}')

    n_steps = 2**np.arange(13, 19) # dt = t_f / n_steps lands exactly on t_f
    fig, ax = plt.subplots(figsize=(6, 4.3))

    for t_f, label, colour, marker in [(0.5*P, r'$t_f = P/2$ (apogee)', BLUE, 'o'),
                                       (P, r'$t_f = P$ (full orbit, perigee)', RED, 's')]:
        dts, errs = [], []
        for n in n_steps:
            dt = t_f/n
            u, _ = rk4(f0, u0, t_f, dt)
            errs.append(np.linalg.norm(u[-1, :2] - kepler_exact_rot(t_f)[:2]))
            dts.append(dt)
        dts, errs = np.array(dts), np.array(errs)
        orders = np.log2(errs[:-1]/errs[1:])
        print(label)
        for dt, err, p in zip(dts, errs*L, np.r_[np.nan, orders]):
            print(f'  dt = {dt:.3e}   error = {err:.3e} km   order = {p:.2f}')
        ax.loglog(dts, errs*L, marker + '-', color=colour, ms=5, lw=1.2, label=label)

    ax.loglog(dts, 3*errs[-1]*L*(dts/dts[-1])**4, 'k--', lw=0.9, label='slope 4 reference')
    ax.set_xlabel(r'time step $\Delta t$ (nondimensional)')
    ax.set_ylabel(r'position error $\varepsilon$ (km)')
    sec = ax.secondary_xaxis('top', functions=(lambda x: x*T, lambda s: s/T))
    sec.set_xlabel(r'$\Delta t$ (s)')
    ax.legend(fontsize=8.5, loc='upper left')
    save(fig, 'verify_kepler')


if __name__ == '__main__':
    main()
