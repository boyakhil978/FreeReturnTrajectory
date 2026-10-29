import numpy as np

from freereturn.constants import MU

def f_spacecraft(u, t, mu=MU):
    """The dynamics of the spacecraft: returns u_dot = f(u, t).

    Parameters
    ----------
    u : ndarray
        The current state [x, y, v_x, v_y]
    t : float
        The current time (unused here — the system is autonomous — but kept in
        the signature so that every marcher can call f the same way)
    mu : float
        Mass ratio m_M / (m_E + m_M)

    Returns
    -------
    u_dot : ndarray
        The four components of du/dt
    """

    r1 = np.sqrt((u[0] + mu)**2 + u[1]**2)
    r2 = np.sqrt((u[0] - 1 + mu)**2 + u[1]**2)

    return np.array([
        u[2],
        u[3],
        (1 - (1 - mu)/r1**3 - mu/r2**3)*u[0] + 2*u[3] + mu*(1 - mu)*(1/r2**3 - 1/r1**3),
        (1 - (1 - mu)/(r1**3) - mu/r2**3)*u[1] - 2*u[2]
    ])

def jacobi_constant(u, mu=MU):
    '''C = 2*Omega - (v_x^2 + v_y^2), conserved along any coasting arc. Works on one state or an array of states.'''
    x, y, vx, vy = u[..., 0], u[..., 1], u[..., 2], u[..., 3]

    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)

    omega = 0.5*(x**2 + y**2) + (1 - mu)/r1 + mu/r2

    return 2*omega - (vx**2 + vy**2)

def jacobian(u, mu=MU):
    '''Jacobian of f with respect to u, evaluated at u.

    Returns
    -------
    J : ndarray, shape (4, 4); J[i, j] = d(u_dot_i) / d(u_j)
    '''

    r1 = np.sqrt((u[0] + mu)**2 + u[1]**2)
    r2 = np.sqrt((u[0] - 1 + mu)**2 + u[1]**2)

    J = np.zeros((4, 4))

    J[0, 2] = 1

    J[1, 3] = 1

    J[2, 0] = (1 - (1 - mu)/r1**3 - mu/r2**3) + (3*(1 - mu)*(u[0] + mu)/(r1**5) + 3*mu*(u[0] - 1 + mu)/(r2**5))*u[0] + mu*(1 - mu)*(3*(u[0] + mu)/(r1**5) - 3*(u[0] - 1 + mu)/(r2**5))
    J[2, 1] = (3*(1 - mu)*u[1]/(r1**5) + 3*mu*u[1]/(r2**5))*u[0] + mu*(1 - mu)*(3*u[1]/(r1**5) - 3*u[1]/(r2**5))
    J[2, 3] = 2

    J[3, 0] = (3*(1 - mu)*(u[0] + mu)/(r1**5) + 3*mu*(u[0] - 1 + mu)/(r2**5))*u[1]
    J[3, 1] = (1 - (1 - mu)/r1**3 - mu/r2**3) + (3*(1 - mu)*u[1]/(r1**5) + 3*mu*u[1]/(r2**5))*u[1]
    J[3, 2] = -2

    return J
