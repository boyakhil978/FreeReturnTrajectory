# Free-Return Trajectory

Models an Earth–Moon free-return trajectory with the planar circular restricted three-body problem (CR3BP), integrated with RK4, and checks the model against Apollo 8 flight data.

## Notebooks

| Notebook | What it covers |
|---|---|
| [`FreeReturnTrajectory.ipynb`](FreeReturnTrajectory.ipynb) | Equations of motion, stiffness (at L4, and why the trajectory's own eigenvalues are what matter), RK4 vs AB4 stability regions, the nominal initial conditions and RK4 self-convergence. |
| [`ApolloComparison.ipynb`](ApolloComparison.ipynb) | Runs the model from Apollo 8's real translunar injection state, grades its predictions against the flight data, and marks where Apollo 8 deliberately left the free return. |
| [`ArtemisIIComparison.ipynb`](ArtemisIIComparison.ipynb) | Verifies the model against orbits from the JPL Three-Body Periodic Orbits catalog, then compares it with Artemis II's flown free return from JPL Horizons, including how well it forecasts the entry flight-path angle. |

## Layout

```
.
├── main.tex                     # report
├── FreeReturnTrajectory.ipynb   # main analysis
├── ApolloComparison.ipynb       # validation against Apollo 8
├── ArtemisIIComparison.ipynb    # verification against the JPL catalog, validation against Artemis II
├── freereturn/                  # shared model code, imported by all notebooks
│   ├── constants.py             # masses, radii, nondimensional scales L, T, V, mu, entry corridor
│   ├── dynamics.py              # f_spacecraft(u, t), its Jacobian, and the Jacobi constant
│   ├── integrators.py           # rk4_step and rk4
│   ├── trajectory.py            # frame conversions, initial states, TLI and burns, perilune / entry detection
│   ├── entry.py                 # fly_to_entry: TLI to entry interface, with an optional midcourse burn
│   ├── comparison.py            # grading and scorecard plot shared by the comparison notebooks
│   ├── jpl.py                   # JPL Horizons and Three-Body Periodic Orbits API clients (cached)
│   ├── apollo8.py               # Apollo 8 reference data, with sources
│   └── artemis2.py              # Artemis II event and burn list, with sources
├── scripts/                     # targeting, verification and time-step studies; each writes to figures/
├── results/nominal.json         # the retargeted nominal, read by the scripts and notebooks
├── figures/                     # figures used in main.tex
├── data/                        # cached JPL API responses, so notebooks run offline
└── tests/
    └── test_model.py            # sanity checks for the model code
```

The notebooks sit at the top level so `import freereturn` works with no install step. Jupyter runs each notebook from its own folder, and that folder holds the package.

## Data

`freereturn/jpl.py` downloads from the [JPL Horizons API](https://ssd-api.jpl.nasa.gov/doc/horizons.html) and the [JPL Three-Body Periodic Orbits API](https://ssd-api.jpl.nasa.gov/doc/periodic_orbits.html) and caches each response as JSON in `data/`. The cache is committed, so every notebook runs offline and gives the same numbers. Delete a file in `data/` to download it again.

## Setup

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Run the tests from the repository root:

```sh
python -m pytest
```

## Reproducing the report figures

Run from the repository root, in this order (about 4 minutes in total):

```sh
python scripts/retarget_nominal.py            # results/nominal.json (run first)
python scripts/make_premise_figure.py         # figures/premise_geometry.pdf
python scripts/verify_kepler.py               # figures/verify_kepler.pdf
python scripts/verify_jpl_orbits.py           # figures/verify_jpl_orbits.pdf
python scripts/verify_jacobi.py               # figures/verify_jacobi.pdf
python scripts/select_dt.py                   # figures/select_dt.pdf
python scripts/stiffness_along_trajectory.py  # figures/stiffness_along_trajectory.pdf
```

| Check | What it tests | Result |
|---|---|---|
| Exact two-body solution (mu = 0, e = 0.97) | Integrator and rotating-frame terms | Observed order 4.02-4.21 |
| JPL catalog orbits (DRO, L1 and L2 Lyapunov), one period | Full three-body equations against published data | Observed order 3.87-4.10; Jacobi constant matches the catalog to 14 digits |
| Jacobi constant along the nominal | Conservation law on the actual trajectory | Drift falls at order 4.03-4.22 |
| Entry flight-path angle vs dt | Event location keeps fourth order; dt selection | Error 2.3e-4 deg at the study dt, about 60x below the tolerances |

## Units

The model works in nondimensional units: length $L$ = 384,400 km (Earth–Moon distance), time $T = \sqrt{L^3/G(m_E + m_M)}$ = 4.3425 days (1/ω of the Earth–Moon system, derived from $L$ and the masses as the nondimensional equations require), and velocity $V = L/T$. Helper functions in `freereturn/trajectory.py` take and return km, km/s and hours.
