import numpy as np

def rk4(f, u_start, t_end, dt):
    '''March u' = f(u, t) from t = 0 to t_end with RK4.

    Returns
    -------
    u : ndarray, shape (n + 1, len(u_start)); row k is the state at t_k
    times : ndarray, the n + 1 times t_k
    '''
    ### YOUR CODE HERE ###

    n = int(round(t_end / dt))
    u = np.empty(((n+1), np.size(u_start)))
    u[0] = u_start

    for k in range(n):
        y1 = f(u[k], k*dt)
        y2 = f(u[k] + 0.5*dt*y1, (k + 0.5)*dt)
        y3 = f(u[k] + 0.5*dt*y2, (k + 0.5)*dt)
        y4 = f(u[k] + dt*y3, (k + 1)*dt)

        u[k+1] = u[k] + (dt/6) * (y1 + 2*y2 + 2*y3 + y4)

    return u, np.arange(n+1)*dt