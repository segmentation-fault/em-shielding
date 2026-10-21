"""T2/T3/T4 -- why the naive bridge stalls, and what the real agreement is.

T1 compares two different physical problems (finite vs infinite shield), so it
cannot yield an error number. Bridging the gap means stretching the circuit
method's plate length L until it is effectively infinite -- but doing only that
does not work, and panel (b) shows it plateauing near 25% instead of converging.

The reason is in _equiv_wire. It conserves the plate's metal volume across both
filament directions, so the mesh presents only

    f = (nx+1) L / [(nx+1) L + (ny+1) W]

of the true sheet conductance to the (essentially unidirectional) current. With
the along-cable spacing dy held fixed, ny+1 ~ L/dy and therefore

    f -> (nx+1) / [(nx+1) + W/dy]

which does not depend on L at all. No amount of length can remove it.

Panel (c) is the test: pooled over every case, the deviation collapses onto
(1 - f). Once f is driven towards 1 with the shield kept long, the circuit
method and FEMM agree to about 1%.

Run fem/compare.py first; this script only plots what it stored.
"""
import os
import sys

import numpy as np

from _common import (C_ISOLATED, C_REF, C_THIRD, C_WELDED, LS_FEMM,  # noqa: E402
                     OUTDIR, length_ramp, plt, save)

RESULTS = os.path.join(os.path.dirname(OUTDIR), "fem", "results.npz")
if not os.path.exists(RESULTS):
    sys.exit(f"{RESULTS} not found -- run:\n    .venv/Scripts/python fem/compare.py")
d = np.load(RESULTS)

x = d["x"]
lengths, curves, dev = d["bridge_lengths"], d["bridge_curves"], d["bridge_dev"]
t4_curves, t4_dev, t4_f = d["t4_curves"], d["t4_dev"], d["t4_fraction"]
t4_par = d["t4_params"]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.0, 4.8))

# ---- (a) profiles ---------------------------------------------------------
for L, curve, color in zip(lengths, curves, length_ramp(len(lengths))):
    ax1.plot(x, curve, color=color, label=f"$L$ = {L:g} m  (fixed mesh)")
best = int(np.argmin(t4_dev[:, 1]))
ax1.plot(x, t4_curves[best], color=C_THIRD, lw=2.2,
         label=f"corrected: $f$={t4_f[best]:.3f}")
ax1.plot(x, d["femm_welded"], color=C_ISOLATED, ls=LS_FEMM, lw=2.0,
         label="FEMM 2-D ($L\\to\\infty$)")
ax1.plot(x, d["analytic"], color=C_REF, lw=1.3, alpha=0.6, label="$B_0$ unshielded")
ax1.set_xlabel("$x$ (m)")
ax1.set_ylabel("$|B|$ ($\\mu$T)")
ax1.set_title("(a) profiles at mid-length")
ax1.set_xlim(x.min(), x.max())
ax1.set_ylim(bottom=0)
ax1.legend(loc="upper right", fontsize=8)

# ---- (b) the stall --------------------------------------------------------
ax2.plot(lengths, dev[:, 0], "o-", color=C_WELDED, label="max")
ax2.plot(lengths, dev[:, 1], "s--", color=C_WELDED, alpha=0.7, label="RMS")
ax2.axhline(dev[1:, 1].mean(), color=C_REF, ls=":", lw=1.2)
ax2.text(lengths[-1], dev[1:, 1].mean(), " plateau", color=C_REF,
         fontsize=9, va="bottom", ha="right")
ax2.set_xscale("log", base=2)
ax2.set_xticks(lengths)
ax2.set_xticklabels([f"{L:g}" for L in lengths])
ax2.set_xlabel("plate length $L$ (m),  mesh fixed")
ax2.set_ylabel("deviation from FEMM 2-D (%)")
ax2.set_title("(b) T2: length alone does not converge\n"
              "($f$ is independent of $L$ at fixed $dy$)")
ax2.set_ylim(bottom=0)
ax2.legend()

# ---- (c) the collapse onto (1 - f) ---------------------------------------
series = [
    ("T2  length sweep", d["bridge_fraction"], dev[:, 1], "o", C_WELDED),
    ("T3  cross-section", d["refine_fraction"], d["refine_dev"][:, 1], "s", C_ISOLATED),
    ("T4  corrected", t4_f, t4_dev[:, 1], "^", C_THIRD),
]
# The L = 4 m case is the genuinely finite 4x4 shield: end-effect dominated, and
# the one point this relation is not claimed to describe. Shown, but hollow, and
# excluded from the fit -- including it would let it drive the slope (0.99 vs 0.64).
outlier = lengths < 8.0
for i, (label, f, rms, marker, color) in enumerate(series):
    f, rms = np.asarray(f), np.asarray(rms)
    mask = outlier if i == 0 else np.zeros(len(f), bool)
    ax3.plot(100.0 * (1.0 - f[~mask]), rms[~mask], marker, color=color,
             ms=9, ls="none", label=label)
    if mask.any():
        ax3.plot(100.0 * (1.0 - f[mask]), rms[mask], marker, mfc="none",
                 mec=color, ms=9, ls="none", mew=1.6)
        ax3.annotate("$L$=4 m: end-effect\ndominated, excluded",
                     (100.0 * (1.0 - f[mask][0]), rms[mask][0]),
                     textcoords="offset points", xytext=(-10, -4),
                     ha="right", va="center", fontsize=8, color=C_REF)

keep = np.concatenate([~outlier, np.ones(len(d["refine_fraction"]), bool),
                       np.ones(len(t4_f), bool)])
all_f = np.concatenate([s[1] for s in series])[keep]
all_rms = np.concatenate([s[2] for s in series])[keep]
shortfall = 100 * (1 - all_f)
slope = float(np.sum(all_rms * shortfall) / np.sum(shortfall ** 2))
grid = np.logspace(np.log10(1.0), np.log10(shortfall.max() * 1.4), 50)
ax3.plot(grid, slope * grid, color=C_REF, ls=":", lw=1.4,
         label=f"RMS $\\approx$ {slope:.2f}$\\,(1-f)$")
# Log-log: the data span 1.1-105% deviation over a 1.7-40% shortfall, so a
# linear axis would compress the whole result into the bottom corner. A
# proportional law appears here as a slope-1 line.
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel("sheet-conductance shortfall  $100\\,(1-f)$  (%)")
ax3.set_ylabel("RMS deviation from FEMM 2-D (%)")
ax3.set_title("(c) the residual gap IS the conductance split")
ax3.legend(fontsize=9, loc="lower right")

fig.tight_layout()
save(fig, "fig_fem_bridge.png")
