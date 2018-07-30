
def get_lmax(cilm, lmax=None):
    """Return truncated coefficients and maximum degree.

    """
    lmax_comp = cilm[0].shape[0] - 1
    if (lmax is not None) and (0 < lmax < lmax_comp):
        cilm = cilm[:, :lmax + 1, :lmax + 1]
        lmax_comp = lmax
    return cilm, lmax_comp

