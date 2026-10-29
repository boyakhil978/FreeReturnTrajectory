import numpy as np

# Physical constants
G = 6.67430e-20 # km^3 / (kg s^2)
M_MOON = 7.3477e22 # kg
M_EARTH = 5.9722e24 # kg
R_EARTH = 6378.165 # km, equatorial radius used in the Apollo reports
R_MOON = 1737.4 # km

# Characteristic units of the CR3BP. The nondimensional equations assume
# G (m_E + m_M) = L^3 / T^2, so T is derived from L rather than set separately.
L = 384400 # km, Earth-Moon distance
T = np.sqrt(L**3 / (G*(M_EARTH + M_MOON))) # s, = 1 / (orbital rate), about 4.3425 days
V = L/T # km/s

MU = M_MOON / (M_MOON + M_EARTH)

# Mission geometry
H_PARK = 185 # km, parking orbit altitude
H_EI = 400000 * 0.3048e-3 # km, entry interface altitude (400,000 ft = 121.92 km)
R_0 = R_EARTH + H_PARK # km
R_EI = R_EARTH + H_EI # km

# Entry corridor (Apollo 13 Flight Journal, Day 5)
GAMMA_TARGET = -6.52 # deg
GAMMA_SHALLOW = -5.25 # deg
GAMMA_STEEP = -7.4 # deg

# Nondimensional versions
R_EARTH_ND = R_EARTH / L
R_MOON_ND = R_MOON / L
R_0_ND = R_0 / L
R_EI_ND = R_EI / L
