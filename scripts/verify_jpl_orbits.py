'''Verification: closure of JPL catalog periodic orbits, as a convergence study.

The two-body test (verify_kepler.py) zeroes every Moon term in f, so it cannot catch a
mistake in them. These catalogued Earth-Moon CR3BP orbits use the full equations: after
exactly one published period each must return to its initial state. A bug in f would make
the closure error level off at a nonzero value instead of falling at fourth order.

Uses the same three orbits as ArtemisIIComparison.ipynb, read from the cached catalog
responses in data/ (JPL Three-Body Periodic Orbits, https://ssd.jpl.nasa.gov/tools/periodic_orbits.html).
Each orbit is integrated with the catalog's own mu.

Run from the repository root:  python scripts/verify_jpl_orbits.py
'''

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent)) # scripts/, for plot_style

from freereturn.dynamics import f_spacecraft, jacobi_constant
from freereturn.integrators import rk4
from freereturn.jpl import periodic_orbits
from plot_style import plt, save, BLUE, RED, GREY

CATALOG_PICKS = [
    ('DRO, 70,000 km (Artemis I size)', 'dro', {'periodmin': 3.1554, 'periodmax': 3.1556}),
    ('L1 Lyapunov', 'lyapunov', {'libr': 1, 'periodmin': 2.9999, 'periodmax': 3.0}),
    ('L2 Lyapunov', 'lyapunov', {'libr': 2, 'periodmin': 3.4498, 'periodmax': 3.4499}),
]


def main():
    fig, ax = plt.subplots(figsize=(6, 4.3))
    n_steps = 2**np.arange(6, 12)

    for (name, family, filters), colour, marker in zip(CATALOG_PICKS, (BLUE, RED, GREY), 'os^'):
        system, orbits = periodic_orbits(family, **filters)
        mu, orbit = system['mu'], orbits[0]
        u0, period = orbit[[0, 1, 3, 4]], orbit[7]
        f = lambda u, t, mu=mu: f_spacecraft(u, t, mu)

        dts = period/n_steps
        errs = np.array([np.linalg.norm(rk4(f, u0, period, dt)[0][-1, :2] - u0[:2]) for dt in dts])*system['lunit']

        print(f'{name}: Jacobi computed {jacobi_constant(u0, mu):.14f}, catalog {orbit[6]:.14f}')
        for dt, err, p in zip(dts, errs, np.r_[np.nan, np.log2(errs[:-1]/errs[1:])]):
            print(f'  dt = {dt:.3e}   closure error = {err:.3e} km   order = {p:.2f}')
        ax.loglog(dts, errs, marker + '-', color=colour, ms=5, lw=1.2, label=name)

    ax.loglog(dts, 3*errs[-1]*(dts/dts[-1])**4, 'k--', lw=0.9, label='slope 4 reference')
    ax.set_xlabel(r'time step $\Delta t$ (nondimensional)')
    ax.set_ylabel(r'closure error $\|\mathbf{r}(P) - \mathbf{r}(0)\|$ (km)')
    ax.legend(fontsize=8, loc='upper left')
    save(fig, 'verify_jpl_orbits')


if __name__ == '__main__':
    main()
