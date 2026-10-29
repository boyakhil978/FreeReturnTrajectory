# Free-Return Trajectory

Models an Earth–Moon free-return trajectory with the planar circular restricted three-body problem (CR3BP), integrated with RK4, and checks the model against Apollo 8 flight data.

## Notebooks

| Notebook | What it covers |
|---|---|
| [`FreeReturnTrajectory.ipynb`](FreeReturnTrajectory.ipynb) | Equations of motion, stiffness analysis at L4, RK4 vs AB4 stability regions, initial conditions and RK4 convergence. |
| [`ApolloComparison.ipynb`](ApolloComparison.ipynb) | Runs the model from Apollo 8's real translunar injection state, grades its predictions against the flight data, and marks where Apollo 8 deliberately left the free return. |

## Layout

```
.
├── FreeReturnTrajectory.ipynb   # main analysis
├── ApolloComparison.ipynb       # validation against Apollo 8
├── freereturn/                  # shared model code, imported by both notebooks
│   ├── constants.py             # masses, radii, nondimensional scales L, T, V, mu
│   ├── dynamics.py              # f_spacecraft(u, t) and its Jacobian
│   ├── integrators.py           # rk4
│   ├── trajectory.py            # frame conversions, initial states, perilune / entry detection
│   └── apollo8.py               # Apollo 8 reference data, with sources
└── tests/
    └── test_model.py            # sanity checks for the model code
```

The notebooks sit at the top level so `import freereturn` works with no install step. Jupyter runs each notebook from its own folder, and that folder holds the package.

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
