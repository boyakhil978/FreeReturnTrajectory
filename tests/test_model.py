import numpy as np

from freereturn.constants import MU, R_0
from freereturn.dynamics import f_spacecraft, jacobian
from freereturn.integrators import rk4
from freereturn.trajectory import earth_dist, initial_state, inertial_speed_fpa

def test_f_is_zero_at_L4():
    u_bar = np.array([0.5 - MU, np.sqrt(3)/2, 0, 0])
    assert np.allclose(f_spacecraft(u_bar, 0), 0, atol=1e-12)

def test_jacobian_matches_finite_differences():
    u = np.array([0.3, -0.4, 0.1, 0.2])
    h = 1e-6

    J_fd = np.zeros((4, 4))
    for j in range(4):
        e = np.zeros(4)
        e[j] = h
        J_fd[:, j] = (f_spacecraft(u + e, 0) - f_spacecraft(u - e, 0)) / (2*h)

    assert np.allclose(jacobian(u, MU), J_fd, atol=1e-6)

def test_rk4_is_fourth_order():
    # u' = i*u has exact solution exp(i*t), the same kind of oscillatory eigenvalue as the CR3BP
    f = lambda u, t: np.array([-u[1], u[0]])
    exact = np.array([np.cos(1), np.sin(1)])

    errors = [np.abs(rk4(f, np.array([1.0, 0.0]), 1, dt)[0][-1] - exact).max() for dt in (0.1, 0.05)]

    assert 15 < errors[0]/errors[1] < 17 # halving dt should cut the error by 2**4

def test_initial_state_round_trip():
    u0 = initial_state(R_0, 10.8, 7.9, 4.2)

    speed, fpa = inertial_speed_fpa(u0)

    assert np.isclose(earth_dist(u0), R_0)
    assert np.isclose(speed, 10.8)
    assert np.isclose(fpa, 7.9)
