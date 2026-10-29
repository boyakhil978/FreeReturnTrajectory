'''Verification 2: RK4 on a published Earth-Moon CR3BP periodic orbit.

The two-body test zeroes every Moon term in f, so it cannot catch a mistake in
them. This test uses the full mu != 0 equations: a distant retrograde orbit
(DRO) from JPL's Three-Body Periodic Orbits catalog must return to its initial
state after exactly one published period. A bug in f would make the closure
error level off at a nonzero value instead of falling at fourth order.

Source: JPL Solar System Dynamics, Three-Body Periodic Orbits,
https://ssd.jpl.nasa.gov/tools/periodic_orbits.html
(API query: sys=earth-moon, family=dro; first entry of the family).
JPL uses its own mu and units (L = 389703.264829278 km, T = 382981.289129055 s),
so this test uses JPL's mu, not the project's.

Run from the repository root:  python scripts/verify_jpl_dro.py
'''

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent)) # scripts/, for plot_style

from Helpers.cr3bp import make_f, jacobi
from Helpers.rk4 import rk4
from plot_style import plt, save, BLUE

MU_JPL = 1.215058560962404e-02
L_JPL = 389703.264829278 # km
U0 = np.array([2.4642189591864819e-02, 0.0, 1.0679956387085975e-12, 7.2237695537238649e+00])
PERIOD = 6.3052152327579369e+00
JACOBI_PUBLISHED = 1.5410005957354


def main():
    print(f'Jacobi constant: computed {jacobi(U0, MU_JPL):.13f}, published {JACOBI_PUBLISHED:.13f}')

    f_jpl = make_f(MU_JPL)
    n_steps = 2**np.arange(14, 19)
    dts, errs = [], []
    for n in n_steps:
        dt = PERIOD/n
        u, _ = rk4(f_jpl, U0, PERIOD, dt)
        errs.append(np.linalg.norm(u[-1, :2] - U0[:2]))
        dts.append(dt)
    dts, errs = np.array(dts), np.array(errs)

    for dt, err, p in zip(dts, errs*L_JPL, np.r_[np.nan, np.log2(errs[:-1]/errs[1:])]):
        print(f'dt = {dt:.3e}   closure error = {err:.3e} km   order = {p:.2f}')

    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.loglog(dts, errs*L_JPL, 'o-', color=BLUE, ms=5, lw=1.2, label='closure error after one period')
    ax.loglog(dts, 3*errs[-1]*L_JPL*(dts/dts[-1])**4, 'k--', lw=0.9, label='slope 4 reference')
    ax.set_xlabel(r'time step $\Delta t$ (nondimensional)')
    ax.set_ylabel(r'$\|\mathbf{r}(P) - \mathbf{r}(0)\|$ (km)')
    ax.legend(fontsize=8.5, loc='upper left')
    save(fig, 'verify_jpl_dro')


if __name__ == '__main__':
    main()
