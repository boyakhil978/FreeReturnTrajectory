'''Download and cache data from JPL's Solar System Dynamics APIs.

- Horizons (https://ssd-api.jpl.nasa.gov/doc/horizons.html): flown spacecraft and Moon ephemerides.
- Three-Body Periodic Orbits (https://ssd-api.jpl.nasa.gov/doc/periodic_orbits.html):
  catalogued CR3BP periodic orbits, used to verify the model independently.

Every fetch is cached as JSON under data/, so the notebooks run offline once the
data has been downloaded and always use the same numbers.
'''

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

HORIZONS_URL = 'https://ssd.jpl.nasa.gov/api/horizons.api'
PERIODIC_ORBITS_URL = 'https://ssd-api.jpl.nasa.gov/periodic_orbits.api'

def _cached_get(url, params, cache_name):
    path = DATA_DIR / cache_name
    if path.exists():
        return json.loads(path.read_text())

    with urllib.request.urlopen(url + '?' + urllib.parse.urlencode(params), timeout=60) as response:
        data = json.loads(response.read())

    DATA_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
    return data

def horizons_vectors(command, start, stop, step, center='500@399'):
    '''Inertial (ICRF) state vectors of a Horizons body relative to `center` (Earth by default).

    Returns
    -------
    jd : ndarray, Julian date (TDB) of each sample
    r : ndarray, shape (n, 3), position [km]
    v : ndarray, shape (n, 3), velocity [km/s]
    '''
    params = {
        'format': 'json',
        'COMMAND': f"'{command}'",
        'EPHEM_TYPE': 'VECTORS',
        'CENTER': f"'{center}'",
        'START_TIME': f"'{start}'",
        'STOP_TIME': f"'{stop}'",
        'STEP_SIZE': f"'{step}'",
        'VEC_TABLE': '2',
        'REF_SYSTEM': 'ICRF',
        'REF_PLANE': 'FRAME',
        'OUT_UNITS': 'KM-S',
        'CSV_FORMAT': 'YES',
        'OBJ_DATA': 'NO',
    }
    cache_name = f'horizons_{command}_{start[:10]}_{stop[:10]}.json'.replace(' ', '_')
    result = _cached_get(HORIZONS_URL, params, cache_name)['result']

    rows = result[result.index('$$SOE') + 5:result.index('$$EOE')].strip().splitlines()
    table = np.array([[float(c) for i, c in enumerate(row.split(',')[:8]) if i != 1] for row in rows])

    return table[:, 0], table[:, 1:4], table[:, 4:7]

def periodic_orbits(family, sys='earth-moon', **filters):
    '''Members of a periodic-orbit family from the JPL Three-Body Periodic Orbits catalog.

    Returns
    -------
    system : dict, the catalog's mass ratio, length unit (km) and time unit (s)
    orbits : ndarray, shape (n, 9), columns x, y, z, vx, vy, vz, jacobi, period, stability
    '''
    params = {'sys': sys, 'family': family, **filters}
    cache_name = 'periodic_orbits_' + '_'.join(f'{v}' for v in params.values()) + '.json'
    data = _cached_get(PERIODIC_ORBITS_URL, params, cache_name)

    system = {'mu': float(data['system']['mass_ratio']),
              'lunit': float(data['system']['lunit']),
              'tunit': float(data['system']['tunit'])}

    return system, np.array(data['data'], dtype=float)
