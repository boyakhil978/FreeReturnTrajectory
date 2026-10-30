'''Q1: sensitivity of gamma_EI to TLI magnitude and timing errors, and the largest errors that
keep the entry inside the corridor.

(a) Magnitude: TLI = nominal + d(dv), phi fixed.
(b) Timing: the burn starts dt_burn seconds late (negative = early). The parking orbit turns at
    n_LEO inertially but the rotating frame turns at 1, so the burn angle from the Earth-Moon
    line shifts by d(phi) = (n_LEO - 1) * dt_burn / T.

Runs that do not reach entry interface on the shallow side skip out of the atmosphere; they are
treated as gamma = 0 in the root finding, which keeps the corridor-edge functions monotonic.

Writes results/q1.json and figures/q1_sensitivity.pdf.
Run from the repository root (after retarget_nominal.py):  python scripts/study_q1.py
'''

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent)) # scripts/, for plot_style

from freereturn.constants import MU, T, R_0_ND, GAMMA_TARGET, GAMMA_SHALLOW, GAMMA_STEEP
from freereturn.entry import entry_angle
from plot_style import plt, save, BLUE, RED, GREY

N_LEO_REL = np.sqrt((1 - MU)/R_0_ND**3) - 1 # parking-orbit rate relative to the rotating frame


def main():
    nominal = json.loads((ROOT/'results'/'nominal.json').read_text())
    dv0, phi0, dt = nominal['delta_v_kms'], nominal['phi_rad'], nominal['dt_nd']

    # gamma_EI as a function of each error; nan = no entry (skips out)
    gamma_mag = lambda e_mms: entry_angle(dv0 + e_mms*1e-6, phi0, dt)
    gamma_time = lambda e_s: entry_angle(dv0, phi0 + N_LEO_REL*e_s/T, dt)

    errors_mag = np.linspace(-20, 20, 81) # mm/s
    errors_time = np.linspace(-2, 2, 81) # s
    g_mag = np.array([gamma_mag(e) for e in errors_mag])
    g_time = np.array([gamma_time(e) for e in errors_time])

    # first-order sensitivities by central differences
    h_mag, h_time = 0.1, 0.005 # mm/s, s
    slope_mag = (gamma_mag(h_mag) - gamma_mag(-h_mag))/(2*h_mag)
    slope_time = (gamma_time(h_time) - gamma_time(-h_time))/(2*h_time)

    # corridor edges: root-find gamma = edge, with skip-outs counted as gamma = 0
    def edge(gamma_fn, errors, gammas, target):
        g = np.nan_to_num(gammas, nan=0.0) - target
        i = np.where(np.sign(g[:-1]) != np.sign(g[1:]))[0]
        return [brentq(lambda e: np.nan_to_num(gamma_fn(e), nan=0.0) - target, errors[j], errors[j + 1], xtol=1e-6)
                for j in i]

    tol = {}
    for name, fn, errs, gs in (('magnitude_mms', gamma_mag, errors_mag, g_mag), ('timing_s', gamma_time, errors_time, g_time)):
        tol[name] = {'shallow_edge': edge(fn, errs, gs, GAMMA_SHALLOW), 'steep_edge': edge(fn, errs, gs, GAMMA_STEEP)}

    results = {
        'slope_deg_per_mms': slope_mag,
        'slope_deg_per_s': slope_time,
        'tolerances': tol,
        'errors_mms': errors_mag.tolist(), 'gamma_mag': g_mag.tolist(),
        'errors_s': errors_time.tolist(), 'gamma_time': g_time.tolist(),
    }
    (ROOT/'results'/'q1.json').write_text(json.dumps(results, indent=2))

    print(f'd gamma / d(dv)    = {slope_mag:+.4f} deg per mm/s')
    print(f'd gamma / d(t_burn) = {slope_time:+.4f} deg per s')
    for name, t in tol.items():
        print(f'{name:>14}: shallow edge at {t["shallow_edge"]}, steep edge at {t["steep_edge"]}')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    panels = ((axes[0], errors_mag, g_mag, tol['magnitude_mms'], r'TLI magnitude error $\delta(\Delta v)$ (mm/s)', '(a) Burn magnitude', 'mm/s'),
              (axes[1], errors_time, g_time, tol['timing_s'], r'TLI timing error $\delta t$ (s)', '(b) Burn timing', 's'))
    for ax, errs, gs, t, xlabel, title, unit in panels:
        ax.axhspan(GAMMA_STEEP, GAMMA_SHALLOW, color='#0ca30c', alpha=0.12, lw=0, label='entry corridor')
        ax.axhline(GAMMA_TARGET, color=GREY, ls='--', lw=0.8, label=r'target $-6.52^\circ$')
        ax.plot(errs, gs, '-', color=BLUE, lw=1.6, label=r'$\gamma_{EI}$')

        skip = np.isnan(gs)
        if skip.any():
            ax.plot(errs[skip], np.zeros(skip.sum()), 'x', color=RED, ms=4, label='skips out (no entry)')

        edges = sorted(t['steep_edge'] + t['shallow_edge'])
        if len(edges) == 2:
            ax.axvspan(edges[0], edges[1], color=GREY, alpha=0.12, lw=0)
            ax.annotate(f'allowed: {edges[0]:+.2f} to {edges[1]:+.2f} {unit}', xy=(0.5, 0.04), xycoords='axes fraction',
                        ha='center', fontsize=8, color='#333333')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'entry flight-path angle $\gamma_{EI}$ (deg)')
        ax.set_title(title, fontsize=10)
        ax.set_ylim(min(np.nanmin(gs), GAMMA_STEEP) - 1, 0.8)
    axes[0].legend(fontsize=8, loc='lower left')
    save(fig, 'q1_sensitivity')


if __name__ == '__main__':
    main()
