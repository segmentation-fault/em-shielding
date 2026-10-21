"""Reproduce Sergeant et al. (COMPEL 2008) Fig. 3:
grid-refinement study - B at P2 = (0, 0, 1) versus the number of filaments along
the shield length n_l, for n_b = 1, 3, 5, parallel and transversal.  Welded shield.
"""
import numpy as np
from _common import (standard_cables, build_shield, profile_points,
                     shielded_norm, save, ZM, plt)

cab = standard_cables()
P2 = profile_points(np.array([0.0]), y=0.0, z=ZM)

n_l_values = list(range(1, 21))
n_b_values = [1, 3, 5]

fig, ax = plt.subplots()
markers = {1: "s", 3: "^", 5: "o"}
for cfg, fill in (("parallel", True), ("transversal", False)):
    for nb in n_b_values:
        B = []
        for nl in n_l_values:
            mesh = build_shield(cfg, n_across=nb, n_along=nl, Rc=0.0)
            b, _ = shielded_norm(mesh, cab, P2)
            B.append(b[0])
        mfc = None if fill else "white"
        ax.plot(n_l_values, B, marker=markers[nb], markersize=5, markerfacecolor=mfc,
                ls="-" if fill else ":",
                label=f"{cfg[:4].capitalize()}., n_b={nb}")

ax.set_xlabel("$n_l$")
ax.set_ylabel("B ($\\mu$T)")
ax.set_title("Sergeant Fig. 3 - grid refinement at P2(0,0,1)")
ax.set_xlim(0, 20)
ax.set_ylim(0, 11)
ax.legend(ncol=2)
save(fig, "fig_sergeant_03.png")
