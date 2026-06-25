"""This module contains frame classes."""

import numpy as _np
from astropy.coordinates import (
    AffineTransform,
    Attribute,
    BaseCoordinateFrame,
    frame_transform_graph,
)
from astropy.coordinates.representation import CartesianRepresentation

from pygeoid.conventions import units as u
from pygeoid.geometry import Position, transform

__all__ = ["LocalFrame", "LocalTangentPlane"]


class LocalFrame(BaseCoordinateFrame):
    """Arbitrary local cartesian frame."""

    default_representation = CartesianRepresentation


class LocalTangentPlane(BaseCoordinateFrame):
    """Local tangent plane geodetic coordiante frame.

    Parameters
    ----------
    *args
        Any representation of the frame data, e.g. x, y, and z coordinates
    origin : `pygeoid.geometry.Position`
        The location on Earth of the local frame origin
    orientation : sequence of str, optional
        The cardinal directions of the x, y, and z axis (default: E, N, U)
    **kwargs
        Any extra BaseCoordinateFrame arguments

    Raises
    ------
    ValueError
        The local frame configuration is not valid

    """

    default_representation = CartesianRepresentation

    origin = Attribute()
    """The origin on Earth of the local frame"""

    orientation = Attribute(default=("E", "N", "U"))
    """The orientation of the local frame, as cardinal directions"""

    def __init__(self, *args, origin, orientation=None, **kwargs):

        if orientation is None:
            super().__init__(*args, origin=origin, **kwargs)
        else:
            super().__init__(*args, origin=origin, orientation=orientation, **kwargs)

        def vector(lat, lon, name):
            _name = name[0].upper()

            azalt = {
                "E": (90, 0),
                "W": (270, 0),
                "N": (0, 0),
                "S": (180, 0),
                "U": (0, 90),
                "D": (0, -90),
            }

            if _name not in azalt:
                raise ValueError(f"Invalid frame orientation `{name}`")

            az, alt = azalt[_name]
            az *= u.deg
            alt *= u.deg

            calt = _np.cos(alt)
            r = [calt * _np.sin(az), calt * _np.cos(az), _np.sin(alt)]
            east, north, up = transform._ecef_to_enu_rotation_matrix(lat, lon)

            d0 = r[0] * east[0] + r[1] * north[0] + r[2] * up[0]
            d1 = r[0] * east[1] + r[1] * north[1] + r[2] * up[1]
            d2 = r[0] * east[2] + r[1] * north[2] + r[2] * up[2]

            return d0, d1, d2

        geodetic = self._origin.geodetic

        ux = vector(geodetic.lat, geodetic.lon, self._orientation[0])
        uy = vector(geodetic.lat, geodetic.lon, self._orientation[1])
        uz = vector(geodetic.lat, geodetic.lon, self._orientation[2])

        self._basis = _np.column_stack((ux, uy, uz))


@frame_transform_graph.transform(AffineTransform, Position, LocalTangentPlane)
def position_to_local(position, local):
    """Compute the transformation from Position to LocalTangentPlane coordinates.

    Parameters
    ----------
    position : Position
        The initial coordinates
    local : LocalTangentPlane
        The LocalTangentPlane frame to transform to

    Returns
    -------
    LocalTangentPlane
        The LocalTangentPlane frame with transformed coordinates
    """
    matrix = local._basis.T
    offset = None
    c = position.represent_as("cartesian")
    if c.x.unit.is_equivalent("m"):
        offset = -local._origin.represent_as("cartesian").transform(matrix)
    return matrix, offset


@frame_transform_graph.transform(AffineTransform, LocalTangentPlane, Position)
def local_to_position(local, position):
    """Compute the transformation from LocalTangentPlane to Position coordinates.

    Parameters
    ----------
    local : LocalTangentPlane
        The initial coordinates in LocalTangentPlane
    position : Position
        The Position frame to transform to

    Returns
    -------
    Position
        The Position frame with transformed coordinates
    """
    matrix = local._basis
    offset = None
    c = local.represent_as("cartesian")
    if c.x.unit.is_equivalent("m"):
        offset = local._origin.represent_as("cartesian")
    return matrix, offset


@frame_transform_graph.transform(AffineTransform, LocalTangentPlane, LocalTangentPlane)
def local_to_local(local0, local1):
    """Compute the transformation between LocalTangentPlane coordinates.

    Parameters
    ----------
    local0 : LocalTangentPlane
        The initial coordinates in the 1st LocalTangentPlane frame.
    local1 : LocalTangentPlane
        The 2nd LocalTangentPlane frame to transform to.

    Returns
    -------
    LocalTangentPlane
        The LocalTangentPlane frame with transformed coordinates.

    """
    matrix = local1._basis.T @ local0._basis
    offset = None
    c = local0.represent_as("cartesian")
    if c.x.unit.is_equivalent("m"):
        offset = (
            local0._origin.represent_as("cartesian")
            - local1._origin.represent_as("cartesian")
        ).transform(local1._basis.T)
    return matrix, offset
