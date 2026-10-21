"""T0 -- the gate that has to pass before any FEM comparison means anything.

With no shield present, FEMM must reproduce the analytic infinite-line source
field. If it does not, the FEMM model (units, phase sequence, boundary, cable
representation) is wrong and every shielded result downstream is noise.

This also settles the open question in README.md: the circuit method gives
B0(0,0,1) = 12.37 uT where the papers show ~14 uT. FEMM is a completely
independent implementation of the same source geometry, so whatever it returns
decides whether that ~12% gap is our bug or the papers' plotting.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem.femm2d import Femm2D                                      # noqa: E402
from shielding.analysis import profile_points, unshielded_norm     # noqa: E402
from shielding.geometry import Cables                              # noqa: E402

TOL = 0.005          # 0.5 %, as specified in the plan

cab = Cables(spacing=0.25, depth=1.5, current=900.0)
x = np.linspace(0.0, 4.0, 21)
pts = profile_points(x, y=0.0, z=1.0)

analytic = unshielded_norm(cab, pts)

with Femm2D(cab, shield=None) as model:
    B_femm = model.sample_B(pts)

from shielding.field import field_norm                             # noqa: E402

femm_uT = field_norm(B_femm) * 1e6

rel = np.abs(femm_uT - analytic) / analytic
rms = float(np.sqrt(np.mean(rel ** 2)))
worst = int(np.argmax(rel))

print("  x (m)   analytic (uT)   FEMM (uT)   rel. dev.")
for xi, a, f, r in zip(x, analytic, femm_uT, rel):
    print(f"  {xi:5.2f}   {a:12.4f}   {f:9.4f}   {r * 100:7.3f} %")

print(f"\nmax deviation {rel.max() * 100:.3f} % at x = {x[worst]:.2f} m")
print(f"RMS deviation {rms * 100:.3f} %")

print(f"\nB0 at P2(0,0,1): analytic {analytic[0]:.2f} uT, FEMM {femm_uT[0]:.2f} uT")
print("(the papers plot ~14 uT here)")

if rel.max() > TOL:
    print(f"\nFAIL: max deviation exceeds {TOL * 100:.1f} %")
    sys.exit(1)
print(f"\nPASS: FEMM agrees with the analytic source to better than {TOL * 100:.1f} %")
