# Free-Return Trajectory Accuracy (AE 370 Group Project 1)

How accurate must a free-return translunar injection (TLI) burn be, and what does correcting an injection error cost? The spacecraft is modelled with the planar circular restricted three-body problem (CR3BP) and integrated with fixed-step RK4, implemented from scratch.

## Layout

| Path | What it is |
|---|---|
| `Helpers/rk4.py` | The numerical method: `rk4_step` and the fixed-step marcher `rk4(f, u_start, t_end, dt)` |
| `Helpers/cr3bp.py` | The dynamics `f(u, t, mu)`, TLI initial state, entry flight-path angle, Jacobi constant, impulsive burns, and `fly_to_entry` (RK4 with event location at entry interface) |
| `Helpers/jacobian.py` | Analytical Jacobian of `f` |
| `Helpers/constants.py` | Physical constants, nondimensional units, entry corridor |
| `scripts/` | One script per study; each writes its figure to `figures/` |
| `results/nominal.json` | Retargeted nominal trajectory, read by the studies and notebooks |
| `FreeReturnTrajectory.ipynb` | Method development: stiffness, stability regions, self-convergence |
| `ApolloComparison.ipynb` | Model validation against Apollo 8 (not a code verification) |
| `main.tex` | The report |

## Reproducing the results

Run everything from the repository root, in this order (about 5 minutes in total):

```bash
pip install -r requirements.txt

python scripts/retarget_nominal.py            # results/nominal.json (run first)
python scripts/make_premise_figure.py         # figures/premise_geometry.pdf
python scripts/verify_kepler.py               # figures/verify_kepler.pdf
python scripts/verify_jpl_dro.py              # figures/verify_jpl_dro.pdf
python scripts/verify_jacobi.py               # figures/verify_jacobi.pdf
python scripts/select_dt.py                   # figures/select_dt.pdf
python scripts/stiffness_along_trajectory.py  # figures/stiffness_along_trajectory.pdf
```

Then run the two notebooks top to bottom.

## Verification summary

| Check | What it tests | Result |
|---|---|---|
| Exact two-body solution (mu = 0, e = 0.97) | Integrator and rotating-frame terms | Observed order 4.02-4.21 |
| JPL Earth-Moon DRO, one period | Full three-body equations against published data | Observed order 4.03-4.13; Jacobi constant matches the catalog to 13 digits |
| Jacobi constant along the nominal | Conservation law on the actual trajectory | Drift falls at order 4.03-4.22 |
| Entry flight-path angle vs dt | Event location keeps fourth order; dt selection | Error 2.3e-4 deg at the study dt, about 60x below the tolerances |

## Sources

- JPL Solar System Dynamics, Three-Body Periodic Orbits: https://ssd.jpl.nasa.gov/tools/periodic_orbits.html
- Apollo 13 Flight Journal, Day 5 (entry corridor): https://apollojournals.org/afj/ap13fj/20day5-wobblesandbursts.html
- Rea, McNamara and Kane, "Orion Artemis I Entry Performance," AAS 24-174 (entry interface at 400 kft)
