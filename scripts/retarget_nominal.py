'''Retarget the nominal free return so that it enters at gamma_EI = -6.52 deg.

phi is held at 4.12 rad and the TLI magnitude is found by bisection. Runs that
do not reach entry interface on the shallow side are treated as gamma = 0
(above the target), so the bisection function is monotonic on the bracket.

Writes results/nominal.json, which every later study reads.
Run from the repository root:  python scripts/retarget_nominal.py
'''

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent)) # scripts/, for plot_style

from freereturn.constants import GAMMA_TARGET, T, MU, V
from freereturn.trajectory import tli_state
from freereturn.entry import fly_to_entry
from freereturn.dynamics import jacobi_constant as jacobi

PHI = 4.12 # rad
DT = 2/2**16 # nondimensional, about 11.4 s; confirmed adequate by select_dt.py
DV_LO, DV_HI = 3.1440, 3.1445 # km/s; shallow (skips out) and steep sides of the target


def miss(delta_v):
    res = fly_to_entry(tli_state(delta_v, PHI), DT)
    gamma = res['gamma_ei'] if res['status'] == 'entry' else 0.0
    return gamma - GAMMA_TARGET


def main():
    lo, hi = DV_LO, DV_HI
    assert miss(lo) > 0 > miss(hi), 'bracket does not straddle the target'

    for _ in range(40):
        mid = 0.5*(lo + hi)
        if miss(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-11:
            break
    delta_v = 0.5*(lo + hi)

    u0 = tli_state(delta_v, PHI)
    res = fly_to_entry(u0, DT)
    nominal = {
        'phi_rad': PHI,
        'delta_v_kms': delta_v,
        'dt_nd': DT,
        'gamma_ei_deg': res['gamma_ei'],
        'v_ei_kms': res['v_ei_kms'],
        't_ei_days': res['t_ei']*T/86400,
        'perilune_alt_km': res['perilune_alt_km'],
        't_perilune_days': res['t_perilune']*T/86400,
        'jacobi_after_tli': float(jacobi(u0)),
        'mu': MU,
        'T_s': T,
        'V_kms': V,
    }

    out = ROOT/'results'
    out.mkdir(exist_ok=True)
    (out/'nominal.json').write_text(json.dumps(nominal, indent=2))

    for key, val in nominal.items():
        print(f'{key:>18}: {val}')


if __name__ == '__main__':
    main()
