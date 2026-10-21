"""J0 -- the 3-D source gate, analogue of tests/verify_fem_b0.py for Elmer.

With no plate present, the Elmer 3-D model must reproduce the analytic
infinite-line source field along the profile. This certifies the whole 3-D
source setup at once: the balanced three-phase currents, their phases, the
finite-length bar closure, the truncation box, and the SaveLine read-out. If it
fails, no shielded 3-D result downstream means anything (same logic as the 2-D
T0 gate).

Unlike the 2-D FEMM gate, the 3-D source has a genuine finite-length
approximation (the cables span +/- box_y, not infinity), so the tolerance here
is looser: the residual is finite-bar-length plus truncation, and it shrinks as
box_y grows. TOL is set where a working point comfortably passes.

Needs Elmer installed (ElmerSolver/ElmerGrid on PATH or $ELMER_HOME) and gmsh
(pip). See fem/elmer3d.py.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem.elmer3d import Cables3D, elmer_norm, analytic_line_B    # noqa: E402
from shielding.field import field_norm                           # noqa: E402

TOL = 0.03            # 3 %, finite-bar + truncation (see module docstring)

cab = Cables3D(spacing=0.25, depth=1.5, current=900.0)
x = np.linspace(0.0, 4.0, 21)
pts = np.column_stack([x, np.zeros_like(x), np.ones_like(x)])

analytic = field_norm(analytic_line_B(cab, pts)) * 1e6
elmer_uT, _ = elmer_norm(cab, pts, with_plate=False)

rel = np.abs(elmer_uT - analytic) / analytic
rms = float(np.sqrt(np.mean(rel ** 2)))
worst = int(np.argmax(rel))

print("  x (m)   analytic (uT)   Elmer (uT)   rel. dev.")
for xi, a, f, r in zip(x, analytic, elmer_uT, rel):
    print(f"  {xi:5.2f}   {a:12.4f}   {f:9.4f}   {r * 100:7.3f} %")

print(f"\nmax deviation {rel.max() * 100:.3f} % at x = {x[worst]:.2f} m")
print(f"RMS deviation {rms * 100:.3f} %")
print(f"\nB0 at P2(0,0,1): analytic {analytic[0]:.2f} uT, Elmer {elmer_uT[0]:.2f} uT")

if rel.max() > TOL:
    print(f"\nFAIL: max deviation exceeds {TOL * 100:.1f} %")
    sys.exit(1)
print(f"\nPASS: Elmer 3-D source agrees with the analytic field to < {TOL * 100:.1f} %")
