'''Eigenvalues of the Jacobian along the nominal free return, and where
lambda * dt lands on the RK4 stability region.

Stiffness has to be judged along the trajectory the method actually integrates.
The largest |lambda| occurs at the low-Earth ends of the trip; it is the
solution's own fast timescale there (accuracy-limited), not a decaying
transient, so the problem is not stiff.

Run from the repository root (after retarget_nominal.py):  python scripts/stiffness_along_trajectory.py
'''

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent)) # scripts/, for plot_style

from freereturn.constants import T, MU
from freereturn.trajectory import tli_state
from freereturn.entry import fly_to_entry
from freereturn.dynamics import jacobian
from plot_style import plt, save, BLUE, RED


def main():
    nominal = json.loads((ROOT/'results'/'nominal.json').read_text())
    dt = nominal['dt_nd']
    res = fly_to_entry(tli_state(nominal['delta_v_kms'], nominal['phi_rad']), dt, store=True)
    u, times = res['u'][::8], res['times'][::8]

    eig = np.array([np.linalg.eigvals(jacobian(uk, MU)) for uk in u])
    lam_max = np.abs(eig).max(axis=1)
    k = np.argmax(lam_max)
    print(f'max |lambda| = {lam_max[k]:.1f} at t = {times[k]*T/86400:.3f} days; |lambda| dt = {lam_max[k]*dt:.4f} (RK4 limit on imaginary axis 2.83)')

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))

    a.semilogy(times*T/86400, lam_max, color=BLUE, lw=1.2)
    a.set_xlabel('time after TLI (days)')
    a.set_ylabel(r'$\max_i |\lambda_i|$ of $\partial \mathbf{f}/\partial \mathbf{u}$')
    a.set_title('(a) Jacobian eigenvalues along the trajectory', fontsize=10)

    x, y = np.meshgrid(np.linspace(-3.2, 1.0, 421), np.linspace(-3.2, 3.2, 641))
    w = x + 1j*y
    R = np.abs(1 + w + w**2/2 + w**3/6 + w**4/24)
    b.contourf(x, y, R <= 1, levels=[0.5, 1.5], colors=['#f4c27a'])
    b.contour(x, y, R, levels=[1], colors='k', linewidths=0.8)
    # Near Earth the eigenvalues are +/- sqrt(2) n (real) and +/- i n, n = local orbital rate.
    # The negative-real and imaginary ones must lie inside the region (stability); the
    # positive-real one is the physical divergence of neighbouring trajectories, which
    # RK4 must resolve accurately rather than damp.
    lam_k = eig[k]
    print('eigenvalues at that state:', np.round(lam_k, 2))
    for dt_show, colour, marker, label in [(dt, RED, 'o', rf'study $\Delta t = {dt:.2e}$'),
                                           (1e-3, BLUE, 's', r'$\Delta t = 10^{-3}$ (stable, but wrong trajectory)')]:
        w_show = lam_k*dt_show
        b.plot(w_show.real, w_show.imag, marker, color=colour, ms=6, mfc='none', mew=1.4, label=label)
    b.annotate(r'$+\sqrt{2}\,n\,\Delta t$: physical divergence' + '\n(resolved for accuracy)',
               xy=(lam_k.real.max()*1e-3, 0), xytext=(0.15, 1.6), fontsize=8,
               arrowprops=dict(arrowstyle='->', lw=0.7))
    b.axhline(0, color='grey', lw=0.6)
    b.axvline(0, color='grey', lw=0.6)
    b.set_aspect('equal')
    b.set_xlim(-3.2, 2.2)
    b.set_xlabel(r'Re$(\lambda \Delta t)$')
    b.set_ylabel(r'Im$(\lambda \Delta t)$')
    b.set_title(r'(b) $\lambda\Delta t$ at the largest $|\lambda|$, on the RK4 region', fontsize=10)
    b.legend(fontsize=7.5, loc='lower left')
    save(fig, 'stiffness_along_trajectory')


if __name__ == '__main__':
    main()
