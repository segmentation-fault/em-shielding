"""Fast (non-FEM) shielding of buried HV cables via the wire-filament circuit method."""
from .geometry import Cables, build_shield, Mesh
from .circuit import solve_currents
from .field import cable_field, biot_savart_segments, field_norm

__all__ = ["Cables", "build_shield", "Mesh", "solve_currents",
           "cable_field", "biot_savart_segments", "field_norm"]
