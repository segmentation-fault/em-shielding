"""Parameter tuning for the 2-D validation -- what actually limits the agreement.

fem/compare.py attributes the whole circuit-vs-FEMM residual to the
sheet-conductance shortfall (1 - f). These six panels test that and find it is
not the dominant term:

  (a) the FEMM reference is not exact either. Refining its own discretization
      moves it by delta_FEM, which is the floor any agreement claim sits on.
  (b) the transverse axis has an OPTIMUM, not a limit. The signed deviation
      crosses zero: too few filaments under-shield (+), too fat filaments
      over-shield (-), because _equiv_wire ties the wire radius to the mesh
      (r ~ 1/sqrt(nx) against a pitch dx ~ 1/nx, so 2r/dx ~ sqrt(nx)).
  (c) the same ladder by RMS, with the f-matched cases from Part B2 landing on
      the same curve -- so the trend is transverse resolution, not f.
  (d) with the cross-section held at its optimum, sweeping 1-f and W/L over a
      decade each barely moves anything: the residual is a constant.
  (e) the two constraint regimes separate cleanly. Isolated plates -- where the
      filament resistance does not matter -- agree to below delta_FEM.
  (f) what the tuning is worth, as profiles.

Run fem/tune.py first; this script only plots what it stored.
"""
import os
import sys

import numpy as np

from _common import (C_ISOLATED, C_REF, C_THIRD, C_WELDED, LS_CIRCUIT,  # noqa: E402
                     LS_FEMM, OUTDIR, length_ramp, plt, save)

RESULTS = os.path.join(os.path.dirname(OUTDIR), "fem", "tune.npz")
if not os.path.exists(RESULTS):
    sys.exit(f"{RESULTS} not found -- run:\n    .venv/Scripts/python fem/tune.py")
d = np.load(RESULTS, allow_pickle=False)

W = 4.0                      # shield width (m), Sergeant Table I -- as in fem/tune.py
DELTA_FEM = float(d["delta_fem"][1])

# Part A needs five categorical slots. These are the remaining Okabe-Ito hues,
# the same colour-vision-safe family _common.py already draws from; validated
# together (worst adjacent CVD dE 11.0 deutan, normal-vision dE 21.1). The two
# lightest carry a sub-3:1 contrast warning, so every line is direct-labelled
# and identity never rests on colour alone.
KNOB_COLORS = [C_WELDED, C_ISOLATED, C_THIRD, "#E69F00", "#CC79A7"]
KNOB_MARKERS = ["o", "s", "^", "D", "v"]

fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.4))
(axA, axB, axC), (axD, axE, axF) = axes


# ---- (a) is the reference ACCURATE, not merely self-converged? -------------
# Self-convergence cannot answer that: the cable-polygon knob converges
# monotonically and away from the truth. With no shield the source is three
# infinite line currents, which shielding.field gives in closed form, so
# accuracy is measurable directly.
cal_p, cal_rms, cal_sd = d["cal_params"], d["cal_rms"], d["cal_sd"]
base = np.isclose(cal_p[:, 2], 5e-3)          # the cable_mesh null duplicate
n_seed = 5.0
for i, arc in enumerate(np.unique(cal_p[base, 0])[::-1]):
    sel = base & np.isclose(cal_p[:, 0], arc)
    order = np.argsort(cal_p[sel, 1])
    axA.errorbar(cal_p[sel, 1][order], cal_rms[sel][order],
                 yerr=cal_sd[sel][order] / np.sqrt(n_seed),
                 marker=KNOB_MARKERS[i], color=KNOB_COLORS[i], ms=7, capsize=3,
                 label=f"arc_seg = {arc:.0f}$\\degree$")
floor = float(d["source_floor"][1])
axA.axhline(floor, color=C_REF, ls="--", lw=1.3)
axA.text(cal_p[base, 1].min(), floor, f" source floor {floor:.2f} %",
         color=C_REF, fontsize=8.5, va="bottom")
axA.axhline(DELTA_FEM, color=C_REF, ls=":", lw=1.2)
axA.text(cal_p[base, 1].min(), DELTA_FEM, f" $\\delta_{{FEM}}$ {DELTA_FEM:.2f} %",
         color=C_REF, fontsize=8.5, va="bottom")
axA.set_xscale("log")
axA.set_xticks(np.unique(cal_p[base, 1]))
axA.set_xticklabels([f"{v:g}" for v in np.unique(cal_p[base, 1])])
axA.minorticks_off()
axA.invert_xaxis()
axA.set_xlabel("air_mesh (m)   $\\longrightarrow$ finer")
axA.set_ylabel("RMS deviation from the exact source (%)")
axA.set_title("(a) FEMM against a closed-form answer\n"
              "finer is not better: the repo defaults already win")
axA.set_ylim(bottom=0)
axA.legend(fontsize=8)

# ---- (b) the transverse ladder, signed ------------------------------------
nac = d["b1_params"][:, 2]
mean, rms1, fat = d["b1_mean"], d["b1_rms"], d["b1_fatness"]
axB.axhline(0.0, color=C_REF, lw=1.2)
axB.axhspan(-DELTA_FEM, DELTA_FEM, color=C_REF, alpha=0.12, lw=0)
axB.plot(nac, mean, "o-", color=C_WELDED, ms=8)
axB.set_xscale("log")
axB.set_xticks(nac)
axB.set_xticklabels([f"{int(n)}" for n in nac])
axB.minorticks_off()
# 2r/dx is the same axis reparametrised (r is tied to nx by _equiv_wire), not a
# second scale -- it is what makes the sign flip legible.
top = axB.twiny()
top.set_xscale("log")
top.set_xlim(axB.get_xlim())
top.set_xticks(nac)
top.set_xticklabels([f"{v:.2f}" for v in fat], fontsize=7.5)
top.minorticks_off()
top.set_xlabel("filament fatness  $2r/dx$", fontsize=9)
lo, hi = axB.get_ylim()
axB.text(nac[0] * 1.05, max(mean) * 0.55,
         "+  under-shielded\n    (too few filaments)", fontsize=8.5, color=C_REF)
axB.text(nac[-1] * 0.98, lo * 0.55, "$-$  over-shielded\n     (filaments overlap)",
         fontsize=8.5, color=C_REF, ha="right")
axB.set_xlabel("$n_{across}$")
axB.set_ylabel("mean signed deviation from FEMM (%)")
axB.set_title("(b) the transverse axis has an optimum\n"
              "two opposite-signed errors, and they cross")

# ---- (c) the same ladder by RMS, plus the f-matched check -----------------
axC.axhspan(0, DELTA_FEM, color=C_REF, alpha=0.12, lw=0)
axC.plot(nac, rms1, "o-", color=C_WELDED, ms=8,
         label=f"$L$={d['b1_params'][0, 0]:g} m, $dy$={d['b1_params'][0, 1]:g} m")
if "b2_params" in d:
    groups = d["b2_group"]
    for g, marker in zip(list(dict.fromkeys(groups.tolist())), ["s", "^"]):
        sel = groups == g
        axC.plot(d["b2_params"][sel, 2], d["b2_rms"][sel], marker, mfc="none",
                 mec=C_ISOLATED, mew=1.6, ms=9, ls="none",
                 label=f"matched {g}")
best_i = int(np.argmin(rms1))
axC.annotate(f"optimum: $n_{{across}}$={int(nac[best_i])}\n"
             f"$2r/dx$={fat[best_i]:.2f}, RMS {rms1[best_i]:.2f} %",
             (nac[best_i], rms1[best_i]), textcoords="offset points",
             xytext=(10, 34), fontsize=8.5, color=C_WELDED,
             arrowprops=dict(arrowstyle="->", color=C_WELDED, lw=1.2))
axC.text(nac[0] * 1.05, DELTA_FEM * 1.15, "$\\delta_{FEM}$", color=C_REF,
         fontsize=8.5)
axC.set_xscale("log")
axC.set_yscale("log")
axC.set_xticks(nac)
axC.set_xticklabels([f"{int(n)}" for n in nac])
axC.minorticks_off()
axC.set_xlabel("$n_{across}$")
axC.set_ylabel("RMS deviation from FEMM (%)")
axC.set_title("(c) refining past the optimum makes it worse\n"
              "(hollow: same sweep at matched $1-f$)")
axC.legend(fontsize=8)

# ---- (d) with the cross-section at its optimum, f and L barely matter -----
b3L = d["b3_params"][:, 0]
uniqL = np.unique(b3L)
u3 = 1.0 - d["b3_f"]
axD.axhspan(-DELTA_FEM, DELTA_FEM, color=C_REF, alpha=0.12, lw=0)
axD.axhline(0.0, color=C_REF, lw=1.0)
FIT_MIN_L = 64.0        # mirrors fem/tune.py; shorter shields are excluded
for L, color in zip(uniqL, length_ramp(len(uniqL))):
    sel = b3L == L
    order = np.argsort(u3[sel])
    # The shortest shield is plotted but hollow: at W/L = 0.125 the circuit
    # method models a genuinely finite shield against an infinite FEMM one, so
    # its swing is end-effect physics, not discretization. Same convention
    # fig_fem_bridge.py uses for its own excluded case.
    excluded = L < FIT_MIN_L
    axD.plot(u3[sel][order], d["b3_mean"][sel][order], "o-", color=color, ms=7,
             mfc="none" if excluded else color,
             mew=1.6 if excluded else 1.0, alpha=0.55 if excluded else 1.0,
             label=f"$L$ = {L:g} m" + (" (excluded)" if excluded else ""))
if "fit_params" in d:
    c, a, alpha, b, beta = d["fit_params"]
    grid = np.logspace(np.log10(u3.min()), np.log10(u3.max()), 60)
    for L, color in zip(uniqL, length_ramp(len(uniqL))):
        axD.plot(grid, c + a * grid ** alpha + b * (W / L) ** beta,
                 color=color, ls=":", lw=1.2)
    axD.axhline(c, color=C_THIRD, ls="--", lw=1.6)
    axD.text(u3.max(), c, f" $c$ = {c:+.2f} %", color=C_THIRD, fontsize=9,
             va="bottom", ha="right")
axD.set_xscale("log")
axD.set_xlabel("conductance shortfall  $1-f$")
axD.set_ylabel("mean signed deviation from FEMM (%)")
axD.set_title("(d) at the optimal cross-section the residual is a constant\n"
              "(dotted: fitted $c + a(1-f)^\\alpha + b(W/L)^\\beta$)")
axD.legend(fontsize=8, ncol=2)

# ---- (e) the two constraint regimes ---------------------------------------
axE.axhspan(0, DELTA_FEM, color=C_REF, alpha=0.12, lw=0)
ny8 = np.isclose(d["b3_params"][:, 1], d["b3_params"][:, 0] / 8.0)
order = np.argsort(d["b3_params"][ny8, 0])
axE.plot(d["b3_params"][ny8, 0][order], d["b3_rms"][ny8][order], "o-",
         color=C_WELDED, ms=8,
         label="welded ($R_c=0$) vs FEMM welded")
if "b4_params" in d:
    o4 = np.argsort(d["b4_params"][:, 0])
    axE.plot(d["b4_params"][:, 0][o4], d["b4_rms"][o4], "s-", color=C_ISOLATED,
             ms=8, label="isolated ($R_c\\to\\infty$) vs FEMM isolated")
axE.text(d["b3_params"][ny8, 0].min(), DELTA_FEM * 1.1,
         "$\\delta_{FEM}$: the reference's own noise", color=C_REF, fontsize=8.5)
axE.set_xscale("log", base=2)
axE.set_xticks(d["b3_params"][ny8, 0][order])
axE.set_xticklabels([f"{L:g}" for L in d["b3_params"][ny8, 0][order]])
axE.minorticks_off()
axE.set_yscale("log")
axE.set_xlabel("plate length $L$ (m)")
axE.set_ylabel("RMS deviation from FEMM (%)")
axE.set_title("(e) both regimes floor at the same $\\approx$0.6 %\n"
              "isolated is length-insensitive; welded needs length to reach it")
axE.legend(fontsize=8)

# ---- (f) what it is worth, as profiles ------------------------------------
x = d["x"]
axF.plot(x, d["ref_converged"], color=C_REF, ls=LS_FEMM, lw=2.2,
         label="FEMM 2-D, converged reference")
worst_i = int(np.argmax(rms1))
axF.plot(x, d["b1_curves"][worst_i], color=C_ISOLATED, ls=LS_CIRCUIT,
         label=f"circuit, $n_{{across}}$={int(nac[worst_i])} "
               f"(RMS {rms1[worst_i]:.1f} %)")
axF.plot(x, d["b1_curves"][best_i], color=C_WELDED, ls=LS_CIRCUIT, lw=2.0,
         label=f"circuit, tuned $n_{{across}}$={int(nac[best_i])} "
               f"(RMS {rms1[best_i]:.2f} %)")
axF.set_xlabel("$x$ (m)")
axF.set_ylabel("$|B|$ ($\\mu$T)")
axF.set_title("(f) the same solver, two parameter choices\n"
              "at $z=1$ m, welded shield")
axF.set_xlim(x.min(), x.max())
axF.set_ylim(bottom=0)
axF.legend(fontsize=8)

fig.tight_layout()
save(fig, "fig_fem_tune.png")
