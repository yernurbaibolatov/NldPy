import numpy as np
from nldpy import solve

class DynamicalSystem:
    def __init__(self, system, t0, x0, params):
        if not callable(system):
            raise TypeError("The 'system' argument must be a callable.")

        # configurations        
        self.dt = 1e-4  # time step for the solver
        self.acc = 1e-8 # values less than acc are considered 0
        self.solver = 'RK45' # ODE solver algorithm

        self.system = system
        self.params = params

        # initial values of the system
        self.t0 = t0
        self.x0 = x0
        self.v0 = self.system(t0, x0, params)

        # current state of the system
        self.t = t0
        self.x = x0
        self.v = self.system(t0, x0, params)

        # trajectory of the latest integration
        self.t_sol = np.zeros(0, dtype=np.float64)
        self.x_sol = np.zeros((0, len(x0)), dtype=np.float64)
        self.v_sol = np.zeros((0, len(x0)), dtype=np.float64)


    def set_state(self, t0, x0):
        """
        Sets the current state of the dynamical system (t, x) 
        to a given position.
        
        The t0 and x0 values are intact, since they are being set at the 
        initialization
        """
        self.t = t0
        self.x = x0
        self.v = self.system(t0, x0, self.params)
    
    def set_parameter(self, p_idx, p_val):
        """
        Sets the value of the parameter with index p_idx to a value p_val
        """
        if p_idx < 0 or p_idx >= len(self.params):
            raise ValueError(f"The p_idx must be within 0 and {len(self.params)-1}")
        
        self.params[p_idx] = p_val

    def reset(self):
        """
        Sets the current state of the dynamical system to the initial 
        position. Parameter values stay intact.

        Useful after an integration.
        """
        pass

    def integrate(self, t_run, t_trans = 0):
        """
        Integrate the system, starting from the current state of
        t and x.

        Parameters:
            t_run: run time of the simulation
            t_trans: transient time of the simulation
        """

        self.t_sol, self.x_sol, self.v_sol = solve(self.system, 
                                                   self.t, self.x, self.params,
                                                   t_run, t_trans,
                                                   self.dt, solver=self.solver)
        
        self.x_sol = self.x_sol.transpose()
        self.v_sol = self.v_sol.transpose()