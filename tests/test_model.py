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

def test_jacobi_constant_matches_jpl_catalog_and_is_conserved():
    from freereturn.dynamics import jacobi_constant
    from freereturn.jpl import periodic_orbits

    system, orbits = periodic_orbits('dro', periodmin=3.1554, periodmax=3.1556)
    mu, orbit = system['mu'], orbits[0]
    u0 = orbit[[0, 1, 3, 4]]

    u, times = rk4(lambda u, t: f_spacecraft(u, t, mu), u0, orbit[7], orbit[7]/2**12)

    assert np.isclose(jacobi_constant(u0, mu), orbit[6], atol=1e-12)
    assert np.abs(jacobi_constant(u, mu) - orbit[6]).max() < 1e-10
    assert np.linalg.norm(u[-1] - u0) < 1e-9 # the catalogued orbit closes after one period

def test_state_from_inertial_rotate_keeps_earth_relative_orbit():
    from freereturn.trajectory import state_from_inertial

    r_moon, v_moon = np.array([384400.0, 0, 0]), np.array([0, 1.0, 0])
    # 20 deg out of the Moon's plane, descending toward Earth
    r = 100000*np.array([0.3, 0.9*np.cos(0.35), 0.9*np.sin(0.35)])
    v = np.array([-1.2, -0.5, 0.2])

    u0 = state_from_inertial(r, v, r_moon, v_moon, method='rotate')
    speed, fpa = inertial_speed_fpa(u0)

    assert np.isclose(earth_dist(u0), np.linalg.norm(r))
    assert np.isclose(speed, np.linalg.norm(v))
    assert np.isclose(fpa, np.degrees(np.arcsin(r @ v / (np.linalg.norm(r)*np.linalg.norm(v)))))
