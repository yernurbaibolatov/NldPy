import numpy as np

def rk45step(system_func, h, t, x, params):
    # Single Runge-Kutta 4 step
    k1 = h * system_func(t, x, params)
    k2 = h * system_func(t + h/2, x + k1/2, params)
    k3 = h * system_func(t + h/2, x + k2/2, params)
    k4 = h * system_func(t + h, x + k3, params)
    x_next = x + k1/6 + k2/3 + k3/3 + k4/6
    return x_next

def solve(system_func, t, x, params, t_run, t_trans=0, dt=1e-4, solver='RK45'):
    """
    Solve the system in a given time span

    Parameters:
        system_func: a callable function
        t: float - initial time (note that it must not be 0)
        x: array(float) - initial state vector of the system
        params: array(float) - parameters of the system
        t_run: float - time duration of the simulation
        t_trans: float - time duration of the transient simulation
        Total simulation time span is: t_run + t_span, starting from t
        dt: single time step of the solver
        solver: the chosen algorithm to solve the ODE
    
    Return:
        An array of time points, an array of state vectors,
        and an array of velocity vectors
        from [t+t_trans] to [t+t_trans+t_run]
    """
    if isinstance(x, float) or isinstance(x, int):
        x = np.array([x])

    t_span = np.arange(t, t+t_trans+t_run, dt)
    x_sol = np.zeros((len(t_span), len(x)), dtype=np.float64)
    v_sol = np.zeros((len(t_span), len(x)), dtype=np.float64)
    x_sol[0] = x
    v_sol[0] = system_func(t, x, params)

    # TODO implement the choice of solvers
    for i in range(1, len(t_span)):
        x_sol[i] = rk45step(system_func, dt, t_span[i-1], x_sol[i-1], params)
        v_sol[i] = system_func(t_span[i], x_sol[i], params)
    
    t_run_sol = t_span[t_span > t + t_trans]
    x_run_sol = x_sol[t_span > t + t_trans]
    v_sun_sol = v_sol[t_span > t + t_trans]

    return t_run_sol, x_run_sol, v_sun_sol

def root_boundaries_1d(system_func, params, a, b, n_brac = 20):
    """
    Finds the brackets [x1, x2] where there is at least one root exists.

    Parameters:
        system_func: a callable function
        params: parameters of the system
        a: left boundary of the search domain
        b: right boundary of the search domain
        n_brac: number of subdomains, in which the roots are searched

    Return:
        array[[xl, xr]]: an array of subdomains (xl, xr)
    """
    if a == b:
        raise ValueError("Parameters a and b should not be equal")
    elif a > b:
        a, b = b, a

    x_domains = np.linspace(a, b, n_brac)
    x_brackets = []

    for i in range(1, len(x_domains)):
        xl, xr = x_domains[i-1], x_domains[i]
        if system_func(0, xl, params)[0]*system_func(0, xr, params)[0] < 0:
            x_brackets.append([xl, xr])
    
    return x_brackets

def bisection_root(system_func, params, a, b, acc):
    """
    Finds the root of the function within [a,b] with given
    accuracy using the bisection method. 
    Needs to be called after the root_boundaries_1d

    Return:
        float: the root of the function with given accuracy:
            |f(x, params)| < acc
    """
    xl, xr = a, b
    x = (xl + xr)/2

    while np.abs(system_func(0, x, params)[0]) > acc:
        if system_func(0, x, params)[0]*system_func(0, xl, params)[0] < 0:
            xl, xr = xl, x
        else:
            xl, xr = x, xr 
        
        x = (xl + xr)/2

    return x