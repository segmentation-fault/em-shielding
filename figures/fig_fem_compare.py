"""T1 -- 2-D FEM (FEMM) against the circuit method on the real 4 m x 4 m shield.

The two curves in each colour pair are *not* expected to coincide, and that is
the point. FEMM's planar model makes the shield infinitely long; the circuit
method models the actual 4 m x 4 m plate array. The gap between dashed and solid
is the finite-length penalty -- exactly the effect that makes 2-D FEM
over-optimistic for this geometry, which is Sergeant et al.'s own criticism of it.

Colour = case (welded/Rc=0 vs isolated/large Rc), linestyle = method
(dashed FEM, solid circuit), so neither is carried by colour alone.

Run fem/compare.py first; this script only plots what it stored.
"""
import os
import sys

import numpy as np

from _common import (C_ISOLATED, C_REF, C_WELDED, LS_CIRCUIT, LS_FEMM,  # noqa: E402
                     OUTDIR, plt, save)

RESULTS = os.path.join(os.path.dirname(OUTDIR), "fem", "results.npz")
if not os.path.exists(RESULTS):
    sys.exit(f"{RESULTS} not found -- run:\n    .venv/Scripts/python fem/compare.py")
d = np.load(RESULTS)

x = d["x"]
fig, ax = plt.subplots(figsize=(7.6, 5.2))

# Drawn as a pale band with the FEM result dotted inside it: the two agree to
# 0.25% RMS, so a same-weight overlay would simply hide one of them.
ax.plot(x, d["analytic"], color=C_REF, lw=4.0, alpha=0.30,
        label="$B_0$ unshielded (analytic)")
ax.plot(x, d["femm_b0"], color=C_REF, ls=":", lw=1.7,
        label="$B_0$ unshielded (FEMM)  [T0 gate, 0.25% RMS]")

ax.plot(x, d["femm_welded"], color=C_WELDED, ls=LS_FEMM,
        label="FEMM 2-D, welded (infinite shield)")
if "circuit_rc0" in d:
    ax.plot(x, d["circuit_rc0"], color=C_WELDED, ls=LS_CIRCUIT,
            label="circuit method, $R_c=0$ (4 m $\\times$ 4 m)")

ax.plot(x, d["femm_isolated"], color=C_ISOLATED, ls=LS_FEMM,
        label="FEMM 2-D, isolated plates ($R_c\\to\\infty$)")
if "circuit_rc100" in d:
    ax.plot(x, d["circuit_rc100"], color=C_ISOLATED, ls=LS_CIRCUIT,
            label="circuit method, $R_c=100$ m$\\Omega$ (4 m $\\times$ 4 m)")

# Annotate the welded gap. It is NOT purely a finite-length effect: at this
# mesh the filament model presents only f = 0.60 of the true sheet conductance
# (see fig_fem_bridge). Stretching L to 32 m closes the gap from ~105% to ~25%
# RMS; the rest is the conductance shortfall.
if "circuit_rc0" in d:
    lo, hi = d["femm_welded"][0], d["circuit_rc0"][0]
    ax.annotate("", xy=(x[0] + 0.05, lo), xytext=(x[0] + 0.05, hi),
                arrowprops=dict(arrowstyle="<->", color=C_WELDED, lw=1.3))
    ax.text(x[0] + 0.14, 0.5 * (lo + hi),
            "finite length\n+ conductance\nshortfall ($f$=0.60)",
            color=C_WELDED, fontsize=8.5, va="center")

ax.set_xlabel("$x$ (m)")
ax.set_ylabel("$|B|$ ($\\mu$T)")
ax.set_title("2-D FEM vs the circuit method at $z=1$ m\n"
             "(dashed = FEM, infinite shield; solid = circuit method, 4 m $\\times$ 4 m)")
ax.set_xlim(x.min(), x.max())
ax.set_ylim(bottom=0)
ax.legend()
save(fig, "fig_fem_compare.png")
