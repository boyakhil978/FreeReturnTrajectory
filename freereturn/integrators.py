import numpy as np

def rk4_step(f, u, t, dt):
    '''One RK4 step of length dt from state u at time t.'''
    y1 = f(u, t)
    y2 = f(u + 0.5*dt*y1, t + 0.5*dt)
    y3 = f(u + 0.5*dt*y2, t + 0.5*dt)
    y4 = f(u + dt*y3, t + dt)

    return u + (dt/6) * (y1 + 2*y2 + 2*y3 + y4)

def rk4(f, u_start, t_end, dt):
    '''March u' = f(u, t) from t = 0 to t_end with RK4.

    Returns
    -------
    u : ndarray, shape (n + 1, len(u_start)); row k is the state at t_k
    times : ndarray, the n + 1 times t_k
    '''
    n = int(round(t_end / dt))
    u = np.empty(((n+1), np.size(u_start)))
    u[0] = u_start

    for k in range(n):
        u[k+1] = rk4_step(f, u[k], k*dt, dt)

    return u, np.arange(n+1)*dt
