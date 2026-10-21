"""Verify that the circuit-method results reproduce the papers' headline claims."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shielding.geometry import Cables, build_shield
from shielding.field import cable_field, field_norm
from shielding.analysis import profile_points, unshielded_norm, shielded_norm

cab = Cables(spacing=0.25, depth=1.5, current=900.0)
P2 = profile_points(np.array([0.0]), y=0.0, z=1.0)
B0 = unshielded_norm(cab, P2)[0]
print(f"Unshielded B0 at P2(0,0,1) = {B0:.2f} uT  (paper ~14 uT)\n")

def sf(Rc, W=4.0, L=4.0, cfg="parallel", n_along=16):
    mesh = build_shield(cfg, W=W, L=L, n_across=3, n_along=n_along, Rc=Rc)
    B, _ = shielded_norm(mesh, cab, P2)
    return B0 / B[0], B[0]

print("Claim 1: welded 4x4 shield gives a few-fold reduction")
s, b = sf(0.0); print(f"  Rc=0    : B={b:.2f} uT, SF={s:.2f}")

print("\nClaim 2 (Sergeant): high contact resistance ~ no better than factor 2")
s, b = sf(0.1); print(f"  Rc=0.1  : B={b:.2f} uT, SF={s:.2f}  (expect ~1-2)")

print("\nClaim 3 (Sergeant): welded 8x8 reduces field ~6x")
s, b = sf(0.0, W=8.0, L=8.0, n_along=32); print(f"  8x8 Rc=0: B={b:.2f} uT, SF={s:.2f}  (expect ~6)")

print("\nClaim 4: parallel vs transversal difference is small")
sp, _ = sf(0.2e-3, cfg="parallel"); st, _ = sf(0.2e-3, cfg="transversal")
print(f"  Rc=0.2mOhm: SF_parallel={sp:.2f}, SF_transversal={st:.2f}")

print("\nClaim 5: contact resistance strongly degrades shielding")
for rc in [0.0, 0.2e-3, 1e-3, 10e-3, 100e-3]:
    s, b = sf(rc); print(f"  Rc={rc*1e3:6.2f} mOhm: B={b:.2f} uT, SF={s:.2f}")
