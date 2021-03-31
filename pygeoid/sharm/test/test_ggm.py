
import os
import tarfile
import tempfile

import numpy as np
import pandas as pd
import astropy.units as u

from pyshtools.shio import read_icgem_gfc
from pygeoid.sharm.ggm import GlobalGravityFieldModel
from pygeoid.sharm.utils import get_lmax
from pygeoid.reduction.normal import LevelEllipsoid
from pygeoid.coordinates.transform import geodetic_to_spherical

ell = LevelEllipsoid('GRS80')

path = os.path.dirname(os.path.abspath(__file__))

def read_test_model():
    model_fname = os.path.join(path, 'data/egm96.gfc.tar.gz')
    temp_path = tempfile.gettempdir()
    tarfile.open(model_fname, "r:gz").extract('egm96.gfc', path=temp_path)
    cnm, gm, r0 = read_icgem_gfc(os.path.join(temp_path, 'egm96.gfc'))
    model = GlobalGravityFieldModel(cnm * u.dimensionless_unscaled,
            gm=gm * u.m**3 / u.s**2, r0=r0 * u.m, ell=ell)
    return model

def read_test_data():
    data_fname = os.path.join(path, 'data/icgem_test_data.csv')
    df = pd.read_csv(data_fname)
    df['latitude'] = df['latitude'].astype(np.float64)
    df['longitude'] = df['longitude'].astype(np.float64)
    df['h_over_geoid'] = df['h_over_geoid'].astype(np.float64)
    return df

model = read_test_model()
data = read_test_data()
latitude = data['latitude'].values * u.deg
longitude = data['longitude'].values * u.deg
radius0 = 0.0 * u.m

def test_get_lmax():
    full_coeffs = model._coeffs.coeffs
    full_coeffs_lmax = full_coeffs[0].shape[0] - 1

    lmax = 180
    truncated_coeffs, truncated_lmax = get_lmax(full_coeffs, lmax)

    truncated_coeffs_lmax = full_coeffs[0].shape[0] - 1

    assert (truncated_coeffs[0].shape[0] - 1) == truncated_lmax
    assert truncated_lmax == lmax

def test_gravitational_potential():
    lat, lon, r = geodetic_to_spherical(latitude, longitude,
            radius0, ell=ell)
    pot_model = model.gravitational_potential.potential(lat, lon, r)
    pot_test = data.potential_ell.values
    np.testing.assert_almost_equal(pot_model.value, pot_test, 6)

def test_gravitation():
    lat, lon, r = geodetic_to_spherical(latitude, longitude,
            radius0, ell=ell)
    gravitation_model = model.gravitation(lat, lon, r).to('mGal')
    gravitation_test = data.gravitation.values
    np.testing.assert_almost_equal(gravitation_model.value, gravitation_test)

def test_gravity():
    lat, lon, r = geodetic_to_spherical(latitude, longitude,
            radius0, ell=ell)
    gravity_model = model.gravity(lat, lon, r).to('mGal')
    gravity_test = data.gravity_ell.values
    np.testing.assert_almost_equal(gravity_model.value,
            gravity_test)

def test_gravity_disturbance_sa():
    lat, lon, r = geodetic_to_spherical(latitude, longitude,
            radius0, ell=ell)
    gravity_dist_model = model.gravity_disturbance_sa(lat, lon, r).to('mGal')
    gravity_dist_test = data.gravity_disturbance_sa.values
    np.testing.assert_almost_equal(gravity_dist_model.value, gravity_dist_test)

def test_gravity_anomaly_sa():
    lat, lon, r = geodetic_to_spherical(latitude, longitude, radius0, ell=ell)
    gravity_anom_model = model.gravity_anomaly_sa(lat, lon, r).to('mGal')
    gravity_anom_test = data.gravity_anomaly_sa.values
    np.testing.assert_almost_equal(gravity_anom_model.value, gravity_anom_test)

def test_height_anomaly_ell():
    lat, lon, r = geodetic_to_spherical(latitude, longitude, radius0, ell=ell)
    ha_model = model.height_anomaly_ell(lat, lon, r)
    ha_test = data.height_anomaly_ell.values
    np.testing.assert_almost_equal(ha_model.value, ha_test)

def test_gradient():
    lat, lon, r = geodetic_to_spherical(latitude, longitude, radius0, ell=ell)
    # rad, lon, lat, total
    gradient_model = model._gravitational.gradient(lat, lon, r)
    r_derivative_model = model._gravitational.r_derivative(lat, lon, r)
    lon_derivative_model = model._gravitational.lon_derivative(lat, lon, r)
    lat_derivative_model = model._gravitational.lat_derivative(lat, lon, r)

    np.testing.assert_almost_equal(gradient_model[0].value,
            r_derivative_model.value)
    np.testing.assert_almost_equal(gradient_model[1].value,
            lon_derivative_model.value)
    np.testing.assert_almost_equal(gradient_model[2].value,
            lat_derivative_model.value)

