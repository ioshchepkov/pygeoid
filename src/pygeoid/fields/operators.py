"""Operators acting on fields."""

__all__ = ["differential", "gradient", "hessian"]


def differential(field, position, *args, **kwargs):
    """Return field differential at a given position."""
    return field.differential(position, *args, **kwargs)


def gradient(field, position, *args, **kwargs):
    """Return field gradient norm at a given position."""
    return field.gradient(position, *args, **kwargs)


def hessian(field, position, *args, **kwargs):
    """Return field Hessian at a given position."""
    return field.hessian(position, *args, **kwargs)
