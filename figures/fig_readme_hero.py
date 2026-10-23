"""README hero image: the (x, z) cross-section, 2-D FEM next to the circuit method.

Left : FEMM 2-D harmonic model -- the welded 8-plate shield, infinitely long.
Right: the circuit method's mid-length slice (y = 0) through a 64 m long shield,
       the T4 configuration of fem/compare.py (dy = 2 m, n_across = 15,
       f = 0.983) that agrees with FEMM to ~1.1% RMS along z = 1 m.

Colour is |B| on a shared log scale; the white lines are the instantaneous field
lines Re(B) at omega*t = 0; the thin contours are |B| iso-lines at the same
levels in both panels, so agreement can be read off directly.

The grids are expensive (a dense 11.8k-unknown solve plus ~30k FEMM point
queries), so they are cached in fem/hero.npz. Pass --recompute to rebuild it;
FEMM 4.2 is only needed then.
"""
import os
import sys

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle

from _common import CURRENT, DEPTH, OUTDIR, SPACING, plt  # noqa: E402
from shielding.circuit import solve_currents                    # noqa: E402
from shielding.field import biot_savart_segments, cable_field   # noqa: E402
from shielding.geometry import Cables, build_shield             # noqa: E402

CACHE = os.path.join(os.path.dirname(OUTDIR), "fem", "hero.npz")

W, N_PLATES, THICK, SIGMA = 4.0, 8, 3e-3, 5.8e7
L_LONG, DY, N_ACROSS = 64.0, 2.0, 15          # T4 best case

# Offsetting the grid by half a step keeps every sample off z = 0, where the
# circuit method's filaments lie and the thin-wire field is singular.
NX, NZ = 241, 131
XG = np.linspace(-4.5, 4.5, NX)
ZG = np.linspace(-2.6, 1.6, NZ) + 0.5 * (4.2 / (NZ - 1))


def grid_points():
    xx, zz = np.meshgrid(XG, ZG)
    return np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()])


def compute():
    cab = Cables(spacing=SPACING, depth=DEPTH, current=CURRENT)
    pts = grid_points()

    print(f"circuit: L={L_LONG:g} m, dy={DY:g} m, n_across={N_ACROSS}")
    mesh = build_shield("parallel", W=W, L=L_LONG, n_plates=N_PLATES,
                        n_across=N_ACROSS, n_along=int(round(L_LONG / DY)),
                        sigma=SIGMA, thickness=THICK)
    I = solve_currents(mesh, cab)
    pa, pb = mesh.segment_endpoints()
    print(f"  {mesh.n_branches} branches solved; Biot-Savart on {len(pts)} points")
    B_circ = cable_field(cab, pts) + biot_savart_segments(pa, pb, I, pts)

    from fem.femm2d import Femm2D, Shield
    print("FEMM 2-D, welded, refined over the whole view")
    shield = Shield(width=W, n_plates=N_PLATES, thickness=THICK, sigma=SIGMA,
                    mode="welded")
    with Femm2D(cab, shield=shield, air_mesh=0.03,
                refine_box=(-4.8, -2.8, 4.8, 1.8)) as m:
        B_femm = m.sample_B(pts)

    shape = (NZ, NX)
    np.savez(CACHE, x=XG, z=ZG, xc=cab.xc, zc=cab.zc,
             femm_bx=B_femm[:, 0].reshape(shape), femm_bz=B_femm[:, 2].reshape(shape),
             circ_bx=B_circ[:, 0].reshape(shape), circ_bz=B_circ[:, 2].reshape(shape))
    print("wrote", CACHE)


def main():
    if "--recompute" in sys.argv or not os.path.exists(CACHE):
        compute()
    d = np.load(CACHE)
    x, z = d["x"], d["z"]

    bg, fg = "#0b0d12", "#e8e8e8"
    norm = LogNorm(vmin=0.5, vmax=2000)
    levels = [1, 3, 10, 30, 100, 300]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.6), sharey=True,
                             facecolor=bg)
    panels = [("femm", "2-D FEM (FEMM)", "infinitely long shield"),
              ("circ", "Circuit method", f"slice at mid-length of a {L_LONG:g} m shield")]
    for ax, (key, title, sub) in zip(axes, panels):
        bx, bz = d[f"{key}_bx"], d[f"{key}_bz"]
        mag = np.sqrt(np.abs(bx) ** 2 + np.abs(bz) ** 2) * 1e6       # uT
        im = ax.pcolormesh(x, z, np.clip(mag, norm.vmin, norm.vmax),
                           norm=norm, cmap="magma", shading="gouraud",
                           rasterized=True)
        cs = ax.contour(x, z, mag, levels=levels, colors="white",
                        linewidths=0.6, alpha=0.35)
        ax.clabel(cs, levels=[3, 30, 300], fmt=lambda v: f"{v:g} µT", fontsize=7,
                  colors="white", inline_spacing=2)
        ax.streamplot(x, z, bx.real, bz.real, color=(0.75, 0.9, 1.0, 0.55),
                      density=1.1, linewidth=0.7, arrowsize=0.6)

        ax.add_patch(Rectangle((-W / 2, -0.035), W, 0.07, facecolor="#7fd3ff",
                               edgecolor="white", lw=0.6, zorder=5))
        for k in range(1, N_PLATES):
            xk = -W / 2 + k * W / N_PLATES
            ax.plot([xk, xk], [-0.035, 0.035], color=bg, lw=0.8, zorder=6)
        for xc, zc, ph in zip(d["xc"], d["zc"], "abc"):
            ax.add_patch(Circle((xc, zc), 0.045, facecolor="#7fd3ff",
                                edgecolor="white", lw=0.8, zorder=5))
            ax.text(xc, zc - 0.12, ph, color=fg, ha="center", va="top",
                    fontsize=8, zorder=6)
        ax.axhline(0, color=fg, lw=0.5, ls=(0, (4, 4)), alpha=0.35, zorder=4)
        ax.plot([x[0], x[-1]], [1.0, 1.0], color="#7fd3ff", lw=0.9,
                ls=(0, (1, 2)), zorder=4)
        ax.text(x[0] + 0.1, 1.06, "$z$ = 1 m", color="#7fd3ff",
                fontsize=8, va="bottom")

        ax.set_title(f"{title}\n", color=fg, fontsize=13, fontweight="bold",
                     loc="left")
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, color="#9aa0aa",
                fontsize=9.5, va="bottom")
        ax.set_aspect("equal")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(z[0], z[-1])
        ax.set_facecolor(bg)
        ax.grid(False)
        ax.tick_params(colors="#9aa0aa", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#3a3f4a")
        ax.set_xlabel("$x$ (m)", color="#9aa0aa")
    axes[0].set_ylabel("$z$ (m)", color="#9aa0aa")

    fig.subplots_adjust(left=0.05, right=0.92, wspace=0.06, top=0.84, bottom=0.17)
    cax = fig.add_axes([0.935, 0.17, 0.012, 0.67])
    fig.text(0.05, 0.035, "Three-phase cables, 900 A at 50 Hz, 1.5 m deep; 8 welded Cu plates (4 m wide, 3 mm) at ground level.  Along $z$ = 1 m the two methods agree to 1.1% RMS.", color="#9aa0aa", fontsize=9.5)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("$|B|$ ($\\mu$T)", color=fg)
    cb.ax.tick_params(colors="#9aa0aa", labelsize=8)
    cb.outline.set_edgecolor("#3a3f4a")

    path = os.path.join(OUTDIR, "fig_readme_hero.png")
    fig.savefig(path, dpi=160, facecolor=bg)
    print("wrote", path)


if __name__ == "__main__":
    main()
