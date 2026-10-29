def jacobian(mu, u):
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