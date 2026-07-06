"""Geometry of the sphere."""

import numpy as np

from pygeoid.conventions import units as u

__all__ = ["Sphere"]


class Sphere:
    """Spherical geometry operations."""

    @staticmethod
    @u.quantity_input
    def spherical_distance(lat1: u.deg, lon1: u.deg, lat2: u.deg, lon2: u.deg):
        """Return angular distances between two collections of points."""
        lat1m, lat2m = np.meshgrid(lat1, lat2)
        lon1m, lon2m = np.meshgrid(lon1, lon2)

        lat_difference = lat1m - lat2m
        lon_difference = lon1m - lon2m

        half_chord = np.sqrt(
            np.sin(0.5 * lat_difference) ** 2
            + np.sin(0.5 * lon_difference) ** 2 * np.cos(lat1m) * np.cos(lat2m)
        )
        return 2 * np.arcsin(half_chord)

    @staticmethod
    @u.quantity_input
    def check_spherical_distance(spherical_distance: u.deg):
        """Validate and return an angular distance."""
        spherical_distance = spherical_distance.to(u.deg)

        if not np.all(
            (0 * u.deg <= spherical_distance) & (spherical_distance <= 180 * u.deg)
        ):
            raise ValueError("Spherical distance values must lie within [0°, 180°]")

        return spherical_distance
