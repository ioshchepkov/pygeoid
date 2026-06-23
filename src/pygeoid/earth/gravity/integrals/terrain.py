"""Terrain related integrals and kernel."""

import numpy.ma as ma

from pygeoid.earth.gravity.integrals.mean_kernel import MeanTerrainCorrectionKernel
from pygeoid.fields.integrals import Integral


class TerrainCorrection(Integral, MeanTerrainCorrectionKernel):
    _name = "Terrain Correction"

    def kernel(self, distance):
        distance = ma.masked_values(distance, 0.0)
        return 1 / distance**3
