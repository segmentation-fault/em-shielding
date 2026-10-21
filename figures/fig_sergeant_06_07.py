"""Reproduce Sergeant et al. (COMPEL 2008) Fig. 6 and Fig. 7 (parametric studies):

Fig. 6  : B(x) at z=1 m for several shield dimensions W x L and Rc in {0, 0.1 Ohm}.
Fig. 7a : B at P2(0,0,1) vs shield width, for several shield lengths l_p.
Fig. 7b : B and B0 at P2(0,0,1) vs cable depth D.
Fig. 7c : B and B0 at P2(0,0,1) vs distance between cables.
"""
import numpy as np
from _common import (build_shield, profile_points, unshielded_norm,
                     shielded_norm, save, ZM, plt, SPACING, DEPTH, CURRENT)
from shielding.geometry import Cables

cab = Cables(spacing=SPACING, depth=DEPTH, current=CURRENT)

# ----------------------------------------------------------------- Fig. 6
x = np.linspace(0, 4, 33)
pts = profile_points(x, y=0.0, z=ZM)
B0 = unshielded_norm(cab, pts)

dims = [(2, 2), (2, 4), (4, 4), (8, 4), (8, 8)]   # (W across x, L along y)
fig, ax = plt.subplots()
ax.plot(x, B0, "k-", lw=2.5, label="B0")
cmap = plt.get_cmap("viridis")
for k, (W, L) in enumerate(dims):
    col = cmap(k / (len(dims) - 1))
    for rc, ls, mk in [(0.1, "--", "s"), (0.0, "-", "o")]:
        nl = int(round(16 * L / 4))
        mesh = build_shield("parallel", W=float(W), L=float(L),
                            n_across=3, n_along=max(4, nl), Rc=rc)
        B, _ = shielded_norm(mesh, cab, pts)
        ax.plot(x, B, ls=ls, color=col, marker=mk, markersize=3,
                markevery=4, label=f"{W}x{L}, Rc={rc:g}")
ax.set_xlabel("x (m)"); ax.set_ylabel("B ($\\mu$T)")
ax.set_title("Sergeant Fig. 6 - shield dimension & contact-resistance sweep")
ax.set_xlim(0, 4); ax.set_ylim(0, 16); ax.legend(ncol=2, fontsize=8)
save(fig, "fig_sergeant_06.png")

# ----------------------------------------------------------------- Fig. 7
P2 = profile_points(np.array([0.0]), y=0.0, z=ZM)
B0_P2 = unshielded_norm(cab, P2)[0]

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))

# (a) B vs shield width, for several shield lengths l_p
widths = np.array([0.5, 1, 2, 3, 4, 6, 8, 10.0])
axs[0].axhline(B0_P2, color="k", ls="-", lw=2, label="$l_p=0$ (no shield)")
for L, mk in [(2, "s"), (4, "^"), (8, "o")]:
    B = []
    for W in widths:
        nl = max(4, int(round(16 * L / 4)))
        mesh = build_shield("parallel", W=float(W), L=float(L),
                            n_across=3, n_along=nl, Rc=0.0)
        b, _ = shielded_norm(mesh, cab, P2)
        B.append(b[0])
    axs[0].plot(widths, B, marker=mk, label=f"$l_p={L}$ m")
axs[0].set_xlabel("shield width (m)"); axs[0].set_ylabel("B ($\\mu$T)")
axs[0].set_title("(a) vs shield width"); axs[0].legend(); axs[0].set_ylim(0, 16)

# (b) B and B0 vs cable depth D (shield fixed at ground, P2 fixed at z=1)
depths = np.linspace(0.3, 4.0, 12)
B0d, Bd = [], []
for D in depths:
    c = Cables(spacing=SPACING, depth=D, current=CURRENT)
    B0d.append(unshielded_norm(c, P2)[0])
    mesh = build_shield("parallel", n_across=3, n_along=16, Rc=0.0)
    b, _ = shielded_norm(mesh, c, P2)
    Bd.append(b[0])
axs[1].semilogy(depths, B0d, "o-", label="B0 (no shield)")
axs[1].semilogy(depths, Bd, "s--", label="B (Rc=0)")
axs[1].set_xlabel("cable depth D (m)"); axs[1].set_ylabel("B ($\\mu$T)")
axs[1].set_title("(b) vs cable depth"); axs[1].legend()

# (c) B and B0 vs distance between cables
spac = np.linspace(0.05, 1.0, 12)
B0s, Bs = [], []
for s in spac:
    c = Cables(spacing=s, depth=DEPTH, current=CURRENT)
    B0s.append(unshielded_norm(c, P2)[0])
    mesh = build_shield("parallel", n_across=3, n_along=16, Rc=0.0)
    b, _ = shielded_norm(mesh, c, P2)
    Bs.append(b[0])
axs[2].plot(spac, B0s, "o-", label="B0 (no shield)")
axs[2].plot(spac, Bs, "s--", label="B (Rc=0)")
axs[2].set_xlabel("distance between cables (m)"); axs[2].set_ylabel("B ($\\mu$T)")
axs[2].set_title("(c) vs cable spacing"); axs[2].legend()

fig.suptitle("Sergeant Fig. 7 - parametric studies at P2(0,0,1)")
fig.tight_layout()
save(fig, "fig_sergeant_07.png")
