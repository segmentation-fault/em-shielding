"""Reproduce De Wulf et al. (JMMM 2007) Fig. 3:
magnetic flux density at 1 m above the six-plate open shield vs horizontal
distance x, for several contact resistances Rcont, via the circuit (MC) method.

De Wulf parameters: line current 570 A, conductor spacing 0.25 m, depth 1.25 m,
six plates 4.00 m x 0.58 m.  The conductive (aluminium-equivalent) material is
modelled with mu_r = 1 and sigma = 30e6 S/m, as stated in the paper.
"""
import numpy as np
from _common import profile_points, unshielded_norm, shielded_norm, save, plt
from shielding.geometry import Cables, build_shield

CURRENT = 570.0
SPACING = 0.25
DEPTH = 1.25
SIGMA = 30e6          # aluminium-equivalent (De Wulf)
THICK = 4e-3
N_PLATES = 6
PLATE_W = 0.58
W = N_PLATES * PLATE_W    # 3.48 m
L = 4.0

cab = Cables(spacing=SPACING, depth=DEPTH, current=CURRENT)
x = np.linspace(0, 4, 41)
pts = profile_points(x, y=0.0, z=1.0)

B0 = unshielded_norm(cab, pts)

fig, ax = plt.subplots()
ax.plot(x, B0, "k-", lw=2.5, label="B0 (unshielded)")

for label, rc, col in [("Rcont=100 mOhm", 100e-3, "tab:blue"),
                       ("Rcont=0.5 mOhm", 0.5e-3, "tab:green"),
                       ("Rcont=0", 0.0, "tab:red")]:
    mesh = build_shield("parallel", W=W, L=L, n_plates=N_PLATES,
                        n_across=3, n_along=16, sigma=SIGMA,
                        thickness=THICK, Rc=rc)
    B, _ = shielded_norm(mesh, cab, pts)
    ax.plot(x, B, color=col, label=label)

ax.set_xlabel("x (m)")
ax.set_ylabel("B ($\\mu$T)")
ax.set_title("De Wulf Fig. 3 - B at 1 m above 6 plates (circuit method)")
ax.set_xlim(0, 4)
ax.set_ylim(0, 9)
ax.legend()
save(fig, "fig_dewulf_03.png")
