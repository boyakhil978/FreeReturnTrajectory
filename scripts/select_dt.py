'''Time-step selection: error in gamma_EI against dt.

The reference is the same trajectory at a much smaller dt. The plot also
checks that the event location keeps fourth-order accuracy. dt is adequate
when the gamma_EI error is well below both
  (i) 1% of the corridor width (2.15 deg), and
  (ii) the change in gamma_EI caused by a 0.1 mm/s TLI error.

Run from the repository root (after retarget_nominal.py):  python scripts/select_dt.py
'''

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent)) # scripts/, for plot_style

from freereturn.constants import T, GAMMA_SHALLOW, GAMMA_STEEP
from freereturn.entry import entry_angle
from plot_style import plt, save, BLUE, RED, GREY


def main():
    nominal = json.loads((ROOT/'results'/'nominal.json').read_text())
    dv, phi, dt_study = nominal['delta_v_kms'], nominal['phi_rad'], nominal['dt_nd']

    dt_ref = 2/2**20
    gamma_ref = entry_angle(dv, phi, dt_ref)

    dts = 2/2**np.arange(13, 19)
    errs = np.array([abs(entry_angle(dv, phi, dt) - gamma_ref) for dt in dts])

    # sensitivity of gamma_EI to TLI magnitude, central difference with 0.1 mm/s
    h = 1e-7 # km/s
    dgamma_ddv = (entry_angle(dv + h, phi, dt_study) - entry_angle(dv - h, phi, dt_study))/(2*h)
    tol_corridor = 0.01*(GAMMA_SHALLOW - GAMMA_STEEP)
    tol_resolution = abs(dgamma_ddv)*1e-7

    print(f'reference gamma_EI = {gamma_ref:.10f} deg at dt = {dt_ref:.3e}')
    print(f'd gamma / d dv = {dgamma_ddv*1e-6:.4f} deg per mm/s')
    print(f'tolerances: 1% of corridor = {tol_corridor:.2e} deg, 0.1 mm/s effect = {tol_resolution:.2e} deg')
    for dt, err, p in zip(dts, errs, np.r_[np.nan, np.log2(errs[:-1]/errs[1:])]):
        flag = '  <- study dt' if np.isclose(dt, dt_study) else ''
        print(f'dt = {dt:.3e} ({dt*T:6.1f} s)   gamma error = {err:.3e} deg   order = {p:.2f}{flag}')

    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.loglog(dts, errs, 'o-', color=BLUE, ms=5, lw=1.2, label=r'$|\gamma_{EI}(\Delta t) - \gamma_{EI}^{ref}|$')
    ax.loglog(dts, 3*errs[-1]*(dts/dts[-1])**4, 'k--', lw=0.9, label='slope 4 reference')
    ax.axhline(tol_corridor, color=GREY, ls=':', lw=1, label='1% of corridor width')
    ax.axhline(tol_resolution, color=RED, ls=':', lw=1, label=r'effect of a 0.1 mm/s TLI error')
    ax.axvline(dt_study, color=GREY, ls='-', lw=0.8)
    ax.text(dt_study, ax.get_ylim()[0]*3, r' study $\Delta t$', fontsize=8, color=GREY)
    ax.set_xlabel(r'time step $\Delta t$ (nondimensional)')
    ax.set_ylabel(r'error in $\gamma_{EI}$ (deg)')
    sec = ax.secondary_xaxis('top', functions=(lambda x: x*T, lambda s: s/T))
    sec.set_xlabel(r'$\Delta t$ (s)')
    ax.legend(fontsize=8, loc='upper left')
    save(fig, 'select_dt')


if __name__ == '__main__':
    main()
