"""High-level helpers: solve a shield and evaluate B along a profile."""

import numpy as np
from .circuit import solve_currents
from .field import cable_field, biot_savart_segments, field_norm


def profile_points(x, y=0.0, z=1.0):
    x = np.asarray(x, float)
    return np.column_stack([x, np.full_like(x, y), np.full_like(x, z)])


def unshielded_norm(cables, points):
    """|B0| (uT) at points, cables only."""
    return field_norm(cable_field(cables, points)) * 1e6


def shielded_norm(mesh, cables, points, freq=50.0):
    """|B| (uT) at points with the shield present."""
    I = solve_currents(mesh, cables, freq=freq)
    pa, pb = mesh.segment_endpoints()
    B = cable_field(cables, points) + biot_savart_segments(pa, pb, I, points)
    return field_norm(B) * 1e6, I
