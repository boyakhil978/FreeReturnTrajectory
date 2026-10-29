# Free-Return Trajectory

Models an Earth–Moon free-return trajectory with the planar circular restricted three-body problem (CR3BP), integrated with RK4, and checks the model against Apollo 8 flight data.

## Notebooks

| Notebook | What it covers |
|---|---|
| [`FreeReturnTrajectory.ipynb`](FreeReturnTrajectory.ipynb) | Equations of motion, stiffness analysis at L4, RK4 vs AB4 stability regions, initial conditions and RK4 convergence. |
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
│   ├── constants.py             # masses, radii, nondimensional scales L, T, V, mu
│   ├── dynamics.py              # f_spacecraft(u, t), its Jacobian, and the Jacobi constant
│   ├── integrators.py           # rk4
│   ├── trajectory.py            # frame conversions, initial states, perilune / entry detection
│   ├── comparison.py            # grading and scorecard plot shared by the comparison notebooks
│   ├── jpl.py                   # JPL Horizons and Three-Body Periodic Orbits API clients (cached)
│   ├── apollo8.py               # Apollo 8 reference data, with sources
│   └── artemis2.py              # Artemis II event and burn list, with sources
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

## Units

The model works in nondimensional units: length $L$ = 384,400 km (Earth–Moon distance), time $T$ = 4.348 days (1/ω of the Earth–Moon system), and velocity $V = L/T$. Helper functions in `freereturn/trajectory.py` take and return km, km/s and hours.
