"""Reproduce Sergeant et al. (COMPEL 2008) Fig. 5:
magnetic flux density along x at z = 1 m above the 8-plate shield, for the
parallel and transversal configurations and contact resistances Rc.
"""
import numpy as np
from _common import (standard_cables, build_shield, profile_points,
                     unshielded_norm, shielded_norm, save, ZM, plt)

cab = standard_cables()
x = np.linspace(0, 4, 41)
pts = profile_points(x, y=0.0, z=ZM)

B0 = unshielded_norm(cab, pts)

RC = [("Rc=0", 0.0), ("Rc=0.2 mOhm", 0.2e-3), ("Rc=100 mOhm", 100e-3)]

fig, ax = plt.subplots()
ax.plot(x, B0, "k-", lw=2.5, label="B0 (unshielded)")

styles = {"parallel": dict(ls="-"), "transversal": dict(ls="--")}
colors = {"Rc=0": "tab:green", "Rc=0.2 mOhm": "tab:red", "Rc=100 mOhm": "tab:blue"}
for cfg in ("parallel", "transversal"):
    for label, rc in RC:
        mesh = build_shield(cfg, n_across=3, n_along=16, Rc=rc)
        B, _ = shielded_norm(mesh, cab, pts)
        ax.plot(x, B, color=colors[label], marker="." if cfg == "parallel" else "",
                markersize=5, label=f"{cfg}, {label}", **styles[cfg])

# approximate 2-D reference: a long welded shield (weak edge effects)
mesh_long = build_shield("parallel", W=4.0, L=20.0, n_across=3, n_along=80, Rc=0.0)
Blong, _ = shielded_norm(mesh_long, cab, pts)
ax.plot(x, Blong, color="gray", ls=":", lw=2, label="long shield (~2D FEM)")

ax.set_xlabel("x (m)")
ax.set_ylabel("B ($\\mu$T)")
ax.set_title("Sergeant Fig. 5 - B at z=1 m above the 8-plate shield")
ax.set_xlim(0, 4)
ax.set_ylim(0, 20)
ax.legend(ncol=2)
save(fig, "fig_sergeant_05.png")
