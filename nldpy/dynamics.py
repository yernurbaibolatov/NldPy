import numpy as np
from nldpy import solve, root_boundaries_1d, bisection_root
import matplotlib.pyplot as plt
from cycler import cycler
import scienceplots

# nice colors for plotting
PLOT_COLORS = [
    '#344965', # Indigo dye
    '#FF6665', # Bittersweet
    '#1D1821', # Rich black
    '#54D6BE', # Turquoise
    '#E5AACE'  # Lavender pink
]


class DynamicalSystem:
    def __init__(self, system, t0, x0, params):
        if not callable(system):
            raise TypeError("The 'system' argument must be a callable.")

        # configurations        
        self.dt = 1e-4  # time step for the solver
        self.acc = 1e-8 # values less than acc are considered 0
        self.solver = 'RK45' # ODE solver algorithm

        self.plot_pts = 1000 # number of points on plots
        plt.style.use(['science', 'nature'])

        plt.rcParams.update({
            # Figure and layout
            'figure.figsize': [12, 6],                 # Default figure size
            'axes.prop_cycle': cycler(color=PLOT_COLORS),  # Custom color cycle for lines
            'lines.linewidth': 2.0,                   # Line width
            'lines.markersize': 8,                    # Marker size
            # Grid
            #'axes.grid': True,                        # Enable grid
            #'grid.alpha': 0.7,                        # Transparency of grid lines
            #'grid.linestyle': '--',                   # Dashed grid lines
            #'grid.linewidth': 0.6,                    # Grid line width

            # Colormap (for plots like heatmaps)
            'image.cmap': 'viridis',                  # Default colormap
            'image.interpolation': 'nearest',         # No smoothing in heatmaps
        })

        # system parameters
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
        self.x_sol = np.zeros(0, dtype=np.float64)
        self.v_sol = np.zeros(0, dtype=np.float64)
        

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


class OneDimensionalDS(DynamicalSystem):
    # 1D system cannot be explicitly time dependent
    def __init__(self, system, x0, params):
        
        def wrapper_func(t, x, p):
            return np.array([system(x, p)])
        super().__init__(wrapper_func, 0, x0, params)

        if not (isinstance(x0, float) or isinstance(x0, int)):
            raise TypeError("Initial condition should be 1D")

        if len(self.system(0, x0, params)) > 1:
            raise TypeError("System return value should be 1D")
        
    def fixed_points(self, a, b):
        """
        Looks for the fixed points of the system
            f(x, params) = 0, for x from a to b
        
        Return:
            array[float] - values of fixed points up to the accuracy
            |f(x, paramas)| < acc
        """

        fp = []
        root_boundaries = root_boundaries_1d(self.system, self.params, 
                                             [a, b], 20)

        for interval in root_boundaries:
            x_root = bisection_root(self.system, self.params,
                                    interval, self.acc)     
                   
            fp.append(x_root)
        
        return fp

    def plot_fixed_points(self, a, b):
        # Plots the flow diagram of the 1D system
        fp = self.fixed_points(a, b)
        x_vals = np.linspace(a, b, self.plot_pts)
        f_vals = self.system(0, x_vals, self.params)[0]

        # Compute directional flow (left or right)
        #flow = np.sign(f_vals)

        y_max = f_vals.max() + 0.1*(f_vals.max()-f_vals.min())
        y_min = f_vals.min() - 0.1*(f_vals.max()-f_vals.min())

        plt.figure()
        plt.axhline(0, lw = 1, linestyle='--', c = PLOT_COLORS[2])
        plt.plot(x_vals, f_vals, c = PLOT_COLORS[1])

        # Mark fixed points
        for p in fp:
            plt.scatter(p, 0, color='red', zorder=3, label='Fixed Point' if
                        'Fixed Point' not in plt.gca().get_legend_handles_labels()[1] else "")

        # Add flow direction arrows
        #for i in range(0, len(x_vals)-1, max(1, self.plot_pts // 10)):
        #    plt.arrow(x_vals[i], 0, 0.8 * flow[i], 0, color=PLOT_COLORS[0],
        #              zorder=4, head_width=0.5, head_length=0.5, lw = 2)


        plt.title("Fixed points and 1D flow diagram")
        plt.xlabel("x")
        plt.ylabel("f(x)")

        plt.xlim(a, b)
        plt.ylim(y_min, y_max)

        plt.legend()

        plt.show()
