'''Verification 3: Jacobi-constant drift along the nominal free return.

C is conserved by the exact coasting CR3BP, so any change along the computed
trajectory is numerical error. Its maximum drift should fall as O(dt^4).

Run from the repository root (after retarget_nominal.py):  python scripts/verify_jacobi.py
'''

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent)) # scripts/, for plot_style

from Helpers.cr3bp import tli_state, fly_to_entry, jacobi
from plot_style import plt, save, BLUE


def main():
    nominal = json.loads((ROOT/'results'/'nominal.json').read_text())
    u0 = tli_state(nominal['delta_v_kms'], nominal['phi_rad'])
    C0 = jacobi(u0)

    dts = 2/2**np.arange(12, 18)
    drifts = []
    for dt in dts:
        res = fly_to_entry(u0, dt, store=True)
        drifts.append(np.abs(jacobi(res['u']) - C0).max())
    drifts = np.array(drifts)

    for dt, d, p in zip(dts, drifts, np.r_[np.nan, np.log2(drifts[:-1]/drifts[1:])]):
        print(f'dt = {dt:.3e}   max |C - C0| = {d:.3e}   order = {p:.2f}')

    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.loglog(dts, drifts, 'o-', color=BLUE, ms=5, lw=1.2, label=r'$\max_k |C(t_k) - C(0)|$, TLI to entry')
    ax.loglog(dts, 3*drifts[-1]*(dts/dts[-1])**4, 'k--', lw=0.9, label='slope 4 reference')
    ax.set_xlabel(r'time step $\Delta t$ (nondimensional)')
    ax.set_ylabel('Jacobi constant drift')
    ax.legend(fontsize=8.5, loc='upper left')
    save(fig, 'verify_jacobi')


if __name__ == '__main__':
    main()
