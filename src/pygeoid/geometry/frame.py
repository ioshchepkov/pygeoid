"""Local Cartesian coordinate containers."""

import numpy as np

from pygeoid.conventions import units as u
from pygeoid.geometry import transform
from pygeoid.geometry.coordinates import CartesianCoordinates

__all__ = ["LocalFrame", "LocalTangentPlane"]


class LocalFrame:
    """Arbitrary local Cartesian frame."""

    def __init__(self, coordinates, y=None, z=None):
        if isinstance(coordinates, CartesianCoordinates):
            self._coordinates = coordinates
        elif y is not None and z is not None:
            self._coordinates = CartesianCoordinates(coordinates, y, z)
        else:
            x, y, z = coordinates
            self._coordinates = CartesianCoordinates(x, y, z)

    @property
    def cartesian(self):
        return self._coordinates

    @property
    def x(self):
        return self.cartesian.x

    @property
    def y(self):
        return self.cartesian.y

    @property
    def z(self):
        return self.cartesian.z

    @property
    def shape(self):
        return np.broadcast_shapes(np.shape(self.x), np.shape(self.y), np.shape(self.z))


class LocalTangentPlane(LocalFrame):
    """Local Cartesian frame tangent to an Earth position."""

    def __init__(self, coordinates=None, *, origin, orientation=("E", "N", "U")):
        self._origin = origin
        self._orientation = orientation
        self._basis = np.column_stack(
            tuple(self._vector(direction) for direction in orientation)
        )
        self._coordinates = coordinates

    @property
    def origin(self):
        return self._origin

    @property
    def orientation(self):
        return self._orientation

    @property
    def cartesian(self):
        if self._coordinates is None:
            raise ValueError("This local tangent plane does not contain a position.")
        return self._coordinates

    def _vector(self, name):
        direction = name[0].upper()
        azalt = {
            "E": (90, 0),
            "W": (270, 0),
            "N": (0, 0),
            "S": (180, 0),
            "U": (0, 90),
            "D": (0, -90),
        }
        try:
            azimuth, altitude = azalt[direction]
        except KeyError as exc:
            raise ValueError(f"Invalid frame orientation `{name}`") from exc

        azimuth *= u.deg
        altitude *= u.deg
        cos_altitude = np.cos(altitude)
        local = np.array(
            [
                cos_altitude * np.sin(azimuth),
                cos_altitude * np.cos(azimuth),
                np.sin(altitude),
            ]
        )

        geodetic = self.origin.geodetic
        ecef_to_enu = transform._ecef_to_enu_rotation_matrix(geodetic.lat, geodetic.lon)
        return local @ ecef_to_enu

    def from_position(self, position):
        """Return this frame containing ``position`` in local coordinates."""
        delta = u.Quantity(
            [
                position.x - self.origin.x,
                position.y - self.origin.y,
                position.z - self.origin.z,
            ]
        )
        x, y, z = self._basis.T @ delta
        return type(self)(
            CartesianCoordinates(x, y, z),
            origin=self.origin,
            orientation=self.orientation,
        )

    def to_position(self):
        """Convert the contained local coordinates to an Earth position."""
        from pygeoid.geometry.position import Position

        local = u.Quantity([self.x, self.y, self.z])
        x, y, z = self._basis @ local + u.Quantity(
            [self.origin.x, self.origin.y, self.origin.z]
        )
        return Position(
            CartesianCoordinates(x, y, z),
            ell=self.origin.ellipsoid,
        )

    def transform_to(self, frame):
        """Transform to another local tangent plane or to ``Position``."""
        from pygeoid.geometry.position import Position

        position = self.to_position()
        if frame is Position:
            return position
        if isinstance(frame, LocalTangentPlane):
            return frame.from_position(position)
        raise TypeError(f"Unsupported target frame: {frame!r}")
