
import numpy as np


class ForwardModel:

    @property
    def density(self):
        """Return density value.

        """
        return self._density

    def gravitation(self, x, y, z):
        """Return gravitation value.

        """
        return np.sqrt(self.gx(x, y, z)**2 + self.gy(x, y, z)**2 +
                       self.gz(x, y, z)**2)

    def tensor(self, x, y, z):
        """Return gradient tensor.

        """
        gxx = self.gxx(x, y, z)
        gyy = self.gyy(x, y, z)
        gzz = self.gzz(x, y, z)
        gxy = self.gxy(x, y, z)
        gxz = self.gxy(x, y, z)
        gyz = self.gxy(x, y, z)

        tensor = np.array([
            [gxx, gxy, gxz],
            [gxy, gyy, gyz],
            [gxz, gyz, gzz]])

        return tensor

    def invariants(self, x, y, z):
        """Return invariants of the gradient tensor.

        """
        tensor = self.tensor(x, y, z)
        i_1 = np.trace(tensor)
        i_2 = 0.5 * (np.trace(tensor)**2 - np.trace(tensor**2))
        i_3 = np.linalg.det(tensor)
        return i_1, i_2, i_3
