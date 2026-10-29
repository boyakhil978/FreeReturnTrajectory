import numpy as np

G = 6.67430e-20 # km^3 / (kg s^2)
M_MOON = 7.3477e22 # kg
M_EARTH = 5.9722e24 # kg
R_EARTH = 6378.165 # km, equatorial radius used in the Apollo reports
R_MOON = 1737.4 # km

# Nondimensional scales. The nondimensional equations assume G (m_E + m_M) = L^3 / T^2,
# so T is derived from L and the masses rather than set separately (about 4.3425 days).
L = 384400 # km
T = np.sqrt(L**3 / (G*(M_EARTH + M_MOON))) # s
V = L/T # km/s

MU = M_MOON / (M_MOON + M_EARTH)

H_PARK = 185 # km, parking-orbit altitude
H_EI = 400000*0.3048e-3 # km, entry interface at 400,000 ft (121.92 km)
R_0 = R_EARTH + H_PARK # km
R_EI = R_EARTH + H_EI # km

# Entry corridor in flight-path angle (Apollo 13 Flight Journal, Day 5)
GAMMA_TARGET = -6.52 # deg
GAMMA_SHALLOW = -5.25 # deg
GAMMA_STEEP = -7.4 # deg

R_0_ND = R_0 / L
R_EI_ND = R_EI / L
R_MOON_ND = R_MOON / L
