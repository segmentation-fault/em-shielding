"""End-to-end sanity checks before generating figures."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shielding.geometry import Cables, build_shield
from shielding.field import cable_field, biot_savart_segments, field_norm
from shielding.circuit import solve_currents

# Sergeant Table I geometry
cab = Cables(spacing=0.25, depth=1.5, current=900.0)

# --- 1. Unshielded field along x at z = 1 m --------------------------------
xs = np.linspace(0, 4, 9)
pts = np.column_stack([xs, np.zeros_like(xs), np.ones_like(xs)])
B0 = field_norm(cable_field(cab, pts)) * 1e6   # uT
print("B0(x) at z=1m [uT]:")
for x, b in zip(xs, B0):
    print(f"  x={x:.2f}  |B0|={b:6.2f}")
print(f"  -> B0(0) = {B0[0]:.2f} uT  (Sergeant Fig.5/6 ~ 14 uT)")

# --- 2. Shielded field, welded (Rc=0), parallel ----------------------------
mesh = build_shield("parallel", n_across=3, n_along=16, Rc=0.0)
print(f"\nMesh: {mesh.n_nodes} nodes, {mesh.n_branches} branches, "
      f"wire r={mesh.radius*1e3:.2f} mm, rho_R={mesh.rho_R:.3e} Ohm/m")
I = solve_currents(mesh, cab, freq=50.0)
pa, pb = mesh.segment_endpoints()
Bsh = cable_field(cab, pts) + biot_savart_segments(pa, pb, I, pts)
Bn = field_norm(Bsh) * 1e6
print("Shielded (parallel, Rc=0):")
for x, b, b0 in zip(xs, Bn, B0):
    sf = b0 / b if b > 0 else float("inf")
    print(f"  x={x:.2f}  |B|={b:6.2f} uT   SF={sf:5.2f}")
print(f"  max induced filament current = {np.abs(I).max():.2f} A")
