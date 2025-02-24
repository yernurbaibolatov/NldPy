"""
# NldPy

NldPy is a Python package for numerical analysis of nonlinear dynamical systems.

## Features
- Euler and Runge-Kutta solvers
- General dynamical system framework
- Phase portrait visualization

## Installation
```bash
pip install .
```

## Example Usage
```python
from nldpy.dynamics import DynamicalSystem
from nldpy.visualization import plot_phase_portrait

def oscillator(t, y):
    return [y[1], -y[0] - 0.1 * y[0]**3]

system = DynamicalSystem(2, oscillator)
t_vals, sol = system.simulate("rk4", [1.0, 0.0], (0, 10), dt=0.01)
plot_phase_portrait(system, (-2, 2), (-2, 2))
```
"""