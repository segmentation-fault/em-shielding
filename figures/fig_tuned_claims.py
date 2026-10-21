"""What the tuned mesh does to the reproduced claims -- and what it cannot fix.

Every paper-reproduction figure here, and the README's "Verified against the
papers" list, runs at Sergeant's standard n_across = 3. fem/tune.py showed that
the transverse axis is the dominant error and found an optimum for it. Applying
that to the papers' own geometry turns up something the long-shield study could
not see.

The trap: a square shield on a square mesh presents exactly half its metal
--------------------------------------------------------------------------
_equiv_wire splits the plate's volume between x- and y-directed filaments, so
the conductance actually offered to a y-directed current is f*sigma*t with

    f = (nx+1) L / [ (nx+1) L + (ny+1) W ]

For W = L and nx = ny this is *identically 1/2*, at every refinement level. So
refining a square shield isotropically converges to a sheet with half the true
conductance, and SF climbs without settling (4x4: 3.6 -> 5.0 and still moving;
8x8: 4.4 -> 11.5). f is not a discretization error that refinement removes; for
this geometry it is a modelling error that refinement preserves.

This is exactly the effect fem/compare.py identifies. It is worth being precise
about where each account applies: in the L >> W limit fem/tune.py works in,
1 - f is ~0.002 and negligible, and the transverse resolution dominates. On the
papers' square shields 1 - f is 0.4-0.5 and dominates everything. Both are true;
neither generalises to the other.

...and there is no mesh that fixes both
--------------------------------------
The only way to raise f is to make the mesh strongly anisotropic: many filaments
across, few along. But a *finite* shield carries closed current loops, and the
along-shield filaments are what carries their return legs. Starve n_along to buy
f and the loops stop resolving, so shielding falls. Measured on the 4x4: driving
1 - f from 0.41 down to 0.01 takes SF from 4.0 down to 1.5 -- more conductance,
less shielding, because the loop resolution collapsed at the same time.

So SF cannot be plotted against 1 - f as if it were a convergence parameter;
the two axes are entangled. Panels (a) and (b) show the actual surface: SF
against n_across, one curve per n_along. The curves cross. At fine n_along
refining the cross-section raises SF; at coarse n_along it lowers it. That is
what non-convergence looks like, and the published point is one arbitrary
location on a surface spanning a factor of two to three.

Panel (c) checks the contact-resistance claims, which do survive, because that
regime is constraint-limited rather than resistance-limited.

Runs its own solves (~6 min); no cached data.
"""
import numpy as np

from _common import (C_ISOLATED, C_REF, C_THIRD, C_WELDED, length_ramp,  # noqa: E402
                     plt,
                     profile_points, save, standard_cables, unshielded_norm)
from shielding.analysis import shielded_norm                     # noqa: E402
from shielding.geometry import build_shield                      # noqa: E402

CAB = standard_cables()
P2 = profile_points(np.array([0.0]), y=0.0, z=1.0)
B0 = unshielded_norm(CAB, P2)[0]
MAX_GB = 20.0

NACS = [3, 9, 21, 35]
NALONGS = [4, 8, 16, 32]
PUBLISHED = {4.0: (3, 16), 8.0: (3, 32)}     # what the repo's figures use


def solve(n_across, n_along, Rc=0.0, W=4.0, L=4.0, n_plates=8):
    mesh = build_shield("parallel", W=W, L=L, n_plates=n_plates,
                        n_across=n_across, n_along=n_along, Rc=Rc)
    n = mesh.n_branches + mesh.n_nodes - 1
    gb = (2 * n ** 2 + 2 * mesh.n_branches ** 2) * 16 / 1024 ** 3
    if gb > MAX_GB:
        print(f"    skip ({gb:.1f} GB)")
        return None
    B, _ = shielded_norm(mesh, CAB, P2)
    nx = n_plates * n_across
    f = (nx + 1) * L / ((nx + 1) * L + (n_along + 1) * W)
    fat = 2.0 * mesh.radius / (W / nx)
    print(f"    {W:g}x{L:g}  n_across={n_across:3d} n_along={n_along:3d}"
          f"  1-f={1 - f:.3f}  2r/dx={fat:.3f}  SF={B0 / B[0]:.2f}")
    return dict(f=f, fat=fat, sf=B0 / B[0], nac=n_across, nal=n_along)


fig, (ax1, axb, ax2) = plt.subplots(1, 3, figsize=(16.4, 5.4))

# ---- (a),(b) the actual surface: SF vs n_across, one curve per n_along ----
for ax, (W, L), tag in ((ax1, (4.0, 4.0), "(a) 4 m $\\times$ 4 m"),
                        (axb, (8.0, 8.0), "(b) 8 m $\\times$ 8 m")):
    print(f"{W:g}x{L:g} welded:")
    for nl, color in zip(NALONGS, length_ramp(len(NALONGS))):
        pts = [(na, solve(na, nl, W=W, L=L)) for na in NACS]
        pts = [(na, r) for na, r in pts if r is not None]
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1]["sf"] for p in pts], "o-",
                color=color, ms=7, label=f"$n_{{along}}$ = {nl}")
        for na, r in pts:
            if (na, nl) == PUBLISHED[W]:
                ax.plot(na, r["sf"], "o", ms=16, mfc="none", mec=C_ISOLATED,
                        mew=2.4, ls="none")
                ax.annotate("published\nhere", (na, r["sf"]),
                            textcoords="offset points", xytext=(16, -22),
                            fontsize=8.5, color=C_ISOLATED)
    ax.axhline(6.0, color=C_THIRD, ls=":", lw=1.5)
    ax.text(NACS[-1], 6.1, 'Sergeant: "up to six times" ', color=C_THIRD,
            fontsize=8.5, va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xticks(NACS)
    ax.set_xticklabels([str(n) for n in NACS])
    ax.minorticks_off()
    ax.set_xlabel("$n_{across}$   $\\longrightarrow$ finer cross-section")
    ax.set_ylabel("shielding factor  $B_0/B$  at $P_2(0,0,1)$")
    ax.set_ylim(0, 15)
    ax.legend(fontsize=8.5)
ax1.set_title("(a) 4 m $\\times$ 4 m welded: SF spans 1.7$-$4.9\n"
              "no plateau along either mesh axis")
axb.set_title("(b) 8 m $\\times$ 8 m welded: SF spans 3.2$-$13.9\n"
              "the \"six times\" match is one point on a surface")

# ---- (c) the contact-resistance claims ------------------------------------
RCS = [0.0, 0.2e-3, 1e-3, 10e-3, 100e-3]
print("contact-resistance sweep:")
xs = np.arange(len(RCS))
for (na, nl), color, marker, tag in (((3, 16), C_REF, "o", "published"),
                                     ((21, 8), C_WELDED, "s", "resolved")):
    sfs = []
    for rc in RCS:
        r = solve(na, nl, Rc=rc)
        sfs.append(np.nan if r is None else r["sf"])
    ax2.plot(xs, sfs, marker=marker, color=color, ms=8,
             ls="-" if tag == "published" else "--",
             label=f"$n_{{across}}$={na}, $n_{{along}}$={nl}  ({tag})")
ax2.axhline(2.0, color=C_THIRD, ls=":", lw=1.5)
ax2.text(xs[-1], 2.05, 'Sergeant: never better than $\\approx 2\\times$ ',
         color=C_THIRD, fontsize=8.5, va="bottom", ha="right")
ax2.set_xticks(xs)
ax2.set_xticklabels(["0\n(welded)", "0.2", "1", "10", "100"])
ax2.set_xlabel("contact resistance $R_c$ (m$\\Omega$)")
ax2.set_ylabel("shielding factor  $B_0/B$  at $P_2(0,0,1)$")
ax2.set_title("(c) the contact-resistance claims do survive\n"
              "that regime is constraint-limited, so the mesh barely matters")
ax2.set_ylim(0, 6)
ax2.legend(fontsize=8.5)

fig.tight_layout()
save(fig, "fig_tuned_claims.png")
