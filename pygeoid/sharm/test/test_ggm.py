
import os
import tarfile
import tempfile

import numpy as np
import pandas as pd

from pyshtools.shio import read_icgem_gfc
from pygeoid.sharm.ggm import GlobalGravityFieldModel
from pygeoid.reduction.normal import LevelEllipsoid
from pygeoid.coordinates.transform import geodetic_to_spherical

ell = LevelEllipsoid('GRS80')

path = os.path.dirname(os.path.abspath(__file__))

def read_test_model():
    model_fname = os.path.join(path, 'data/egm96.gfc.tar.gz')
    temp_path = tempfile.gettempdir()
    tarfile.open(model_fname, "r:gz").extract('egm96.gfc', path=temp_path)
    cnm, gm, r0 = read_icgem_gfc(os.path.join(temp_path, 'egm96.gfc'))
    model = GlobalGravityFieldModel(cnm, gm=gm, r0=r0, ell=ell)
    return model

def read_test_data():
    data_fname = os.path.join(path, 'data/icgem_test_data.csv')
    df = pd.read_csv(data_fname)
    df['latitude'] = df['latitude'].astype(np.float)
    df['longitude'] = df['longitude'].astype(np.float)
    df['h_over_geoid'] = df['h_over_geoid'].astype(np.float)
    return df

model = read_test_model()
data = read_test_data()
latitude = data['latitude'].values
longitude = data['longitude'].values

def test_gravitational_potential():
    lat, lon, r = geodetic_to_spherical(latitude, longitude,
            0.0, ell=ell, degrees=True)
    pot_model = model.gravitational_potential.potential(lat, lon, r, degrees=True)
    pot_test = data.potential_ell.values
    np.testing.assert_almost_equal(pot_model, pot_test, 6)

def test_gravitation():
    lat, lon, r = geodetic_to_spherical(latitude, longitude,
            0.0, ell=ell, degrees=True)
    gravitation_model = model.gravitation(lat, lon, r, degrees=True) * 1e5
    gravitation_test = data.gravitation.values
    np.testing.assert_almost_equal(gravitation_model, gravitation_test)

def test_gravity():
    lat, lon, r = geodetic_to_spherical(latitude, longitude,
            0.0, ell=ell, degrees=True)
    gravity_model = model.gravity(lat, lon, r, degrees=True) * 1e5
    gravity_test = data.gravity_ell.values
    np.testing.assert_almost_equal(gravity_model, gravity_test)

def test_gravity_disturbance_sa():
    lat, lon, r = geodetic_to_spherical(latitude, longitude, 0.0, ell=ell,
            degrees=True)
    gravity_dist_model = model.gravity_disturbance_sa(lat, lon, r, degrees=True) * 1e5
    gravity_dist_test = data.gravity_disturbance_sa.values
    np.testing.assert_almost_equal(gravity_dist_model, gravity_dist_test)

def test_gravity_anomaly_sa():
    lat, lon, r = geodetic_to_spherical(latitude, longitude, 0.0, ell=ell,
            degrees=True)
    gravity_anom_model = model.gravity_anomaly_sa(lat, lon, r, degrees=True) * 1e5
    gravity_anom_test = data.gravity_anomaly_sa.values
    np.testing.assert_almost_equal(gravity_anom_model, gravity_anom_test)

def test_height_anomaly_ell():
    lat, lon, r = geodetic_to_spherical(latitude, longitude, 0.0, ell=ell,
            degrees=True)
    ha_model = model.height_anomaly_ell(lat, lon, r, degrees=True)
    ha_test = data.height_anomaly_ell.values
    np.testing.assert_almost_equal(ha_model, ha_test)
