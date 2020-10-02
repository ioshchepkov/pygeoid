# -*- coding: utf-8 -*-
"""Terrain related integrals and kernel.

"""

from pygeoid.integrals.core import Integral
from pygeoid.integrals.mean_kernel import MeanTerrainCorrectionKernel


class TerrainCorrection(Integral, MeanTerrainCorrectoinKernel):

    _name = 'Terrain Correction'

    def kernel(self, distance):
        distance = ma.masked_values(distance, 0.0)
        return 1 / distance**3
