
import os
import tarfile
import tempfile

import numpy as np
import pandas as pd
import astropy.units as u

from pyshtools.shio import read_icgem_gfc
from pygeoid.sharmonics.ggm import GlobalGravityFieldModel
from pygeoid.sharmonics.utils import get_lmax
from pygeoid.potential.normal import LevelEllipsoid
from pygeoid.coordinates.frame import ECEF


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

position = ECEF.from_geodetic(
        data['latitude'].values * u.deg,
        data['longitude'].values * u.deg,
        0.0 * u.m, ell=ell)

def test_get_lmax():
    full_coeffs = model._coeffs.coeffs
    full_coeffs_lmax = full_coeffs[0].shape[0] - 1

    lmax = 180
    truncated_coeffs, truncated_lmax = get_lmax(full_coeffs, lmax)

    truncated_coeffs_lmax = full_coeffs[0].shape[0] - 1

    assert (truncated_coeffs[0].shape[0] - 1) == truncated_lmax
    assert truncated_lmax == lmax

def test_gravitational_potential():
    pot_model = model.gravitational_potential(position)
    pot_test = data.potential_ell.values
    np.testing.assert_almost_equal(pot_model.value, pot_test, 6)

def test_gravitation():
    gravitation_model = model.gravitation(position).to('mGal')
    gravitation_test = data.gravitation.values
    np.testing.assert_almost_equal(gravitation_model.value, gravitation_test)

def test_gravity():
    gravity_model = model.gravity(position).to('mGal')
    gravity_test = data.gravity_ell.values
    np.testing.assert_almost_equal(gravity_model.value,
            gravity_test)

def test_gravity_disturbance_sa():
    gravity_dist_model = model.gravity_disturbance_sa(position).to('mGal')
    gravity_dist_test = data.gravity_disturbance_sa.values
    np.testing.assert_almost_equal(gravity_dist_model.value, gravity_dist_test)

def test_gravity_anomaly_sa():
    gravity_anom_model = model.gravity_anomaly_sa(position).to('mGal')
    gravity_anom_test = data.gravity_anomaly_sa.values
    np.testing.assert_almost_equal(gravity_anom_model.value, gravity_anom_test)

def test_height_anomaly_ell():
    ha_model = model.height_anomaly_ell(position)
    ha_test = data.height_anomaly_ell.values
    np.testing.assert_almost_equal(ha_model.value, ha_test)

def test_dov():

    position = ECEF.from_geodetic(
            data['latitude'].values * u.deg,
            data['longitude'].values * u.deg,
            data['h_over_geoid'].values * u.m,
            ell=ell)

    dov_ew_model = model.vertical_deflection_ew(
            position).to('arcsec').value
    dov_ns_model = model.vertical_deflection_ns(
            position).to('arcsec').value
    dov_abs_model = model.vertical_deflection_abs(
            position).to('arcsec').value

    dov_ew_test = data.dov_ew.values
    dov_ns_test = data.dov_ns.values
    dov_abs_test = data.dov_abs.values

    np.testing.assert_almost_equal(dov_ew_model, dov_ew_test, 3)
    np.testing.assert_almost_equal(dov_ns_model, dov_ns_test, 3)
    np.testing.assert_almost_equal(dov_abs_model, dov_abs_test, 3)

def test_r_derivatives():
    r_derivative_model = model._gravitational.r_derivative(position)
    r_derivative_differential_model = model._gravitational.differential(
            position).d_distance

    np.testing.assert_almost_equal(r_derivative_model.value,
            r_derivative_differential_model.value)

def test_lat_derivatives():
    lat_derivative_model = model._gravitational.lat_derivative(
            position) / position.spherical.distance**2 * u.rad
    lat_derivative_differential_model = model._gravitational.differential(
            position).d_lat

    np.testing.assert_almost_equal(lat_derivative_model.value,
            lat_derivative_differential_model.value)

def test_lon_derivatives():
    lon_derivative_model = model._gravitational.lon_derivative(
            position) / position.spherical.distance**2 * u.rad
    lon_derivative_differential_model = model._gravitational.differential(
            position).d_lon

    np.testing.assert_almost_equal(lon_derivative_model.value,
            lon_derivative_differential_model.value)
