"""The reproduced paper profiles, with the mesh uncertainty they actually carry.

figures/fig_sergeant_05.py and figures/fig_dewulf_03.py each draw one curve per
contact resistance, at Sergeant's standard n_across = 3. fig_tuned_claims.py
shows that this geometry has no converged mesh in the (n_across, n_along)
family: for a square shield the conductance fraction f is structural, not a
discretization error, and buying f back by starving n_along destroys the
in-plane current loops instead.

So the honest "tuned" version of these figures is not a different single curve.
It is the same curve with a band: every mesh in a defensible range, drawn
together. Where the band is thin the published result is trustworthy; where it
is fat, the single published curve was never determined by the physics alone.

Read it this way:

  * the low-Rc bands are wide (18-27%) -- those cases are resistance-limited, so
    they inherit the full f ambiguity;
  * the high-Rc bands collapse to ~4% -- constraint-limited, mesh-insensitive,
    and those are the claims that survive;
  * B0 is exact (a closed-form line-current sum) and carries no band at all.

Runs its own solves (~4 min); no cached data.
"""
import numpy as np

from _common import (C_ISOLATED, C_REF, C_THIRD, C_WELDED, ZM,  # noqa: E402
                     plt, profile_points, save, standard_cables,
                     unshielded_norm)
from shielding.analysis import shielded_norm                    # noqa: E402
from shielding.geometry import Cables, build_shield             # noqa: E402

# The mesh family. Chosen to bracket what a careful user might pick: the
# published mesh, a refined cross-section, and the anisotropic meshes that
# chase f. All are defensible; that is exactly the problem.
MESHES = [(3, 16), (9, 16), (21, 16), (21, 8)]
PUBLISHED = (3, 16)

X = np.linspace(0, 4, 41)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.6))


def band(ax, cases, geom, title, ylim):
    """Draw one uncertainty band per contact resistance."""
    cab = geom["cables"]
    pts = profile_points(X, y=0.0, z=geom["z"])
    ax.plot(X, unshielded_norm(cab, pts), color=C_REF, lw=2.6,
            label="$B_0$ unshielded (exact)")

    for (label, rc), color in zip(cases, (C_WELDED, C_THIRD, C_ISOLATED)):
        curves, published = [], None
        for nac, nal in MESHES:
            mesh = build_shield("parallel", W=geom["W"], L=geom["L"],
                                n_plates=geom["n_plates"], n_across=nac,
                                n_along=nal, sigma=geom["sigma"],
                                thickness=geom["thick"], Rc=rc)
            B, _ = shielded_norm(mesh, cab, pts)
            curves.append(B)
            if (nac, nal) == PUBLISHED:
                published = B
            print(f"    {label:<16s} n_across={nac:3d} n_along={nal:3d}"
                  f"  B(0)={B[0]:6.3f} uT")
        C = np.array(curves)
        lo, hi = C.min(axis=0), C.max(axis=0)
        spread = 100.0 * np.median((hi - lo) / np.maximum(published, 1e-9))
        ax.fill_between(X, lo, hi, color=color, alpha=0.25, lw=0)
        ax.plot(X, published, color=color, lw=1.9,
                label=f"{label}  (mesh spread {spread:.0f}%)")

    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$|B|$ ($\\mu$T)")
    ax.set_title(title)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, ylim)
    ax.legend(fontsize=8.5)


print("Sergeant Fig. 5 geometry:")
band(ax1,
     [("$R_c=0$ (welded)", 0.0), ("$R_c=0.2$ m$\\Omega$", 0.2e-3),
      ("$R_c=100$ m$\\Omega$", 100e-3)],
     dict(cables=standard_cables(), W=4.0, L=4.0, n_plates=8, sigma=5.8e7,
          thick=3e-3, z=ZM),
     "(a) Sergeant Fig. 5, parallel\n"
     "line = as published ($n_{across}$=3); band = the mesh family",
     16.0)

print("De Wulf Fig. 3 geometry:")
band(ax2,
     [("$R_{cont}=0$", 0.0), ("$R_{cont}=0.5$ m$\\Omega$", 0.5e-3),
      ("$R_{cont}=100$ m$\\Omega$", 100e-3)],
     dict(cables=Cables(spacing=0.25, depth=1.25, current=570.0),
          W=6 * 0.58, L=4.0, n_plates=6, sigma=30e6, thick=4e-3, z=1.0),
     "(b) De Wulf Fig. 3\n"
     "same pattern: low $R_c$ 18$-$20% wide, high $R_c$ 4%",
     9.0)

fig.tight_layout()
save(fig, "fig_tuned_profiles.png")
