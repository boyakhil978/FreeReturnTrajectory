'''Artemis II reference data. Times are hours after the end of the TLI burn.

Sources
-------
- JPL Horizons, spacecraft -1024 (Artemis II / Orion "Integrity"): flown trajectory from NASA/JSC
  navigation ephemerides, and the mission event list in the Horizons object data.
- JPL Horizons, body 301 (Moon): lunar ephemeris over the same span.
'''

from datetime import datetime, timedelta

HORIZONS_ID = -1024

# Horizons returns TDB; the event list is UTC. TDB - UTC = 69.184 s in 2026.
TDB_MINUS_UTC = timedelta(seconds=69.184)

TLI_END_UTC = datetime(2026, 4, 2, 23, 54)

# Span of the flown trajectory used here: from just after TLI until just before entry interface
HORIZONS_START = '2026-04-03 00:30' # TDB
HORIZONS_STOP = '2026-04-10 23:50' # TDB
HORIZONS_STEP = '5 min'

def utc_to_hours(utc):
    return (utc - TLI_END_UTC).total_seconds() / 3600

T_DATA_START = utc_to_hours(datetime(2026, 4, 3, 0, 30) - TDB_MINUS_UTC)

# Event list (Horizons object data)
artemis2 = {
    'tli_dv': 388, # m/s, from a 2223 x 70377 km high Earth orbit
    'closest_approach_moon_centre': 8282, # km
    't_closest_approach': utc_to_hours(datetime(2026, 4, 6, 23, 1)),
    'max_earth_dist': 413146.2, # km
    't_max_earth_dist': utc_to_hours(datetime(2026, 4, 6, 23, 5)),
    't_ei': utc_to_hours(datetime(2026, 4, 10, 23, 53)), # 122 km altitude
}

# Every burn after TLI: (name, hours after TLI, delta-v in m/s, kind)
#   trim     - small trajectory correction, path stays on the free return
#   entry    - crew-module burn after service-module separation, sets the final entry angle
artemis2_burns = [
    ('OTC-3', utc_to_hours(datetime(2026, 4, 6, 3, 3, 17)), 3.0, 'trim'),
    ('RTC-1', utc_to_hours(datetime(2026, 4, 8, 0, 3)), 0.49, 'trim'),
    ('RTC-2', utc_to_hours(datetime(2026, 4, 10, 2, 53)), 1.62, 'trim'),
    ('RTC-3', utc_to_hours(datetime(2026, 4, 10, 18, 53)), 1.28, 'trim'),
    ('CM raise', utc_to_hours(datetime(2026, 4, 10, 23, 36)), 3.0, 'entry'),
]
