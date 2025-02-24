import numpy as np

def rk45step(system_func, h, t, x, params):
    # Single Runge-Kutta 4 step
    k1 = h * system_func(t, x, params)
    k2 = h * system_func(t + h/2, x + k1/2, params)
    k3 = h * system_func(t + h/2, x + k2/2, params)
    k4 = h * system_func(t + h, x + k3, params)
    x_next = x + k1/6 + k2/3 + k3/3 + k4/6
    return x_next

def solve(system_func, t, x, params, t_run, t_trans=0, dt=1e-3, solver='RK45'):
    """
    Solve the system in a given time span

    Parameters:
        t: float - initial time (note that it must not be 0)
        x: array(float) - initial state vector of the system
        params: array(float) - parameters of the system
        t_run: float - time duration of the simulation
        t_trans: float - time duration of the transient simulation
        Total simulation time span is: t_run + t_span, starting from t
        dt: single time step of the solver
        solver: the chosen algorithm to solve the ODE
    
    Return:
        An array of time points, and an array of state vectors
        from [t+t_trans] to [t+t_trans+t_run]
    """
    t_span = np.arange(t, t+t_trans+t_run, dt)
    x_sol = np.zeros((len(t_span), len(x)), dtype=np.float64)
    x_sol[0] = x
