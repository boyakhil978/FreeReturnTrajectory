'''Apollo 8 trajectory data, converted to km, km/s and hours after TLI.

Sources
-------
- R. W. Orloff, Apollo by the Numbers, NASA SP-2000-4029: Translunar, Lunar Orbit and
  Transearth phase tables, and mission narrative.
- NASA, Apollo 8 Press Kit (Release 68-208), 1968: planned free-return entry time.
- NASA, Apollo 8 Mission Report (MSC-PA-R-69-1): entry flight-path angle.
'''

from freereturn.constants import R_EARTH, R_MOON, R_EI

NMI = 1.852 # km
FT = 0.3048e-3 # km

def get_to_hours(h, m, s):
    '''Ground elapsed time (from liftoff) in hours.'''
    return h + m/60 + s/3600

GET_TLI = get_to_hours(2, 56, 5.51)

apollo8 = {
    # Translunar injection (Apollo by the Numbers, Translunar Phase table)
    'tli_alt': 187.221*NMI, # km
    'tli_speed': 35505.41*FT, # km/s, space fixed
    'tli_fpa': 7.897, # deg, space fixed

    # Closest approach to the Moon on the uncorrected free-return trajectory
    'perilune_alt_free_return': 458.1*NMI, # km
    # LOI ignition is just before pericynthion, so use it as the arrival time
    't_perilune': get_to_hours(69, 8, 20.4) - GET_TLI, # h after TLI
    'loi_alt': 75.6*NMI, # km
    # Lunar pass after MCC-1 lowered it for LOI
    'perilune_alt_after_mcc1': 66.3*NMI, # km

    # Entry interface, 400,000 ft (actual flight, after the lunar orbit stay)
    'ei_speed': 36221.1*FT, # km/s
    'ei_fpa': -6.50, # deg (Mission Report)
    # Press kit: on a pure free return, entry "at about 136 hours after liftoff"
    't_ei_free_return': 136 - GET_TLI, # h after TLI
}

# Every burn Apollo 8 made after TLI: (name, hours after TLI, delta-v in ft/s, kind)
#   trim      - a few ft/s, path stays essentially on the free return
#   retarget  - deliberately moved the lunar pass off the free-return value
#   departure - left the free-return trajectory entirely
apollo8_burns = [
    ('Separation 1', get_to_hours(3, 40, 1) - GET_TLI, 1.1, 'trim'),
    ('Separation 2', get_to_hours(4, 45, 1) - GET_TLI, 7.7, 'trim'),
    ('MCC-1', get_to_hours(10, 59, 59.2) - GET_TLI, 20.4, 'retarget'),
    ('MCC-2', get_to_hours(60, 59, 55.9) - GET_TLI, 1.4, 'trim'),
    ('LOI', get_to_hours(69, 8, 20.4) - GET_TLI, 2997, 'departure'),
    ('TEI', get_to_hours(89, 19, 16.6) - GET_TLI, 3519.0, 'departure'),
    ('MCC-5', get_to_hours(104, 0, 0) - GET_TLI, 4.8, 'trim'),
]

T_MCC1 = apollo8_burns[2][1]
T_LOI = apollo8_burns[4][1]

def phase(t):
    '''Which part of the flight a measurement at t hours after TLI comes from.'''
    if t <= T_MCC1:
        return 'free return'
    if t < T_LOI:
        return 'after MCC-1'
    return 'after LOI'

# Measured distances: (name, hours after TLI, 'earth' or 'moon', km from that body's centre)
apollo8_points = [
    ('TLI', 0, 'earth', R_EARTH + 187.221*NMI),
    ('CSM separation', get_to_hours(3, 20, 59.3) - GET_TLI, 'earth', R_EARTH + 3797.775*NMI),
    ('MCC-1 ignition', T_MCC1, 'earth', R_EARTH + 52768.4*NMI),
    ('Gravity crossover', get_to_hours(55, 38, 40) - GET_TLI, 'earth', 176250*NMI),
    ('Gravity crossover', get_to_hours(55, 38, 40) - GET_TLI, 'moon', 33800*NMI),
    ('MCC-2 ignition', get_to_hours(60, 59, 55.9) - GET_TLI, 'moon', R_MOON + 21064.5*NMI),
    ('LOI ignition', T_LOI, 'moon', R_MOON + apollo8['loi_alt']),
    ('TEI ignition', get_to_hours(89, 19, 16.6) - GET_TLI, 'moon', R_MOON + 60.2*NMI),
    ('Entry interface', get_to_hours(146, 46, 12.8) - GET_TLI, 'earth', R_EI),
]
