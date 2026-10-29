'''Planar circular restricted three-body model of an Earth–Moon free-return trajectory.

Modules
-------
constants   : physical constants and nondimensional scales
dynamics    : equations of motion f(u, t) and their Jacobian
integrators : time marchers (RK4)
trajectory  : frame conversions, initial states, TLI and burns, event detection
entry       : TLI to entry interface with fixed-step RK4 and an optional midcourse burn
apollo8     : Apollo 8 reference data for validation
'''
