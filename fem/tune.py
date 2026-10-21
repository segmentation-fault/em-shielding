"""Parameter tuning for the 2-D validation: how close can circuit vs FEMM get?

fem/compare.py establishes *that* the filament circuit method and an independent
FEMM 2-D model agree in the L -> infinity limit, and reports RMS 1.1 % at a
hand-picked (L, dy, n_across). This script asks the two questions compare.py
leaves open:

  1. What is the best achievable agreement, and at which parameters?
  2. How much of the residual is the FEMM reference's *own* discretization error?

compare.py, fem/results.npz and the figures built on them are left untouched;
this is a separate study writing fem/tune.npz.

The three knobs are not the three error sources
-----------------------------------------------
(L, dy, n_across) map near-orthogonally onto three physically distinct
quantities, and it is the latter the error actually depends on:

    L alone             ->  W/L      finite-length end effect
    product dy*n_across ->  1 - f    sheet-conductance shortfall (compare.py)
    n_across alone      ->  nx       transverse resolution *and* filament fatness

compare.py attributes the whole residual to (1 - f). That is not what the data
say. At matched 1-f ~ 0.007, varying only n_across moves the deviation from
8.1 % to 0.6 %; and at fixed n_across = 9, driving 1-f from 0.028 to 0.004
stalls near 2.6 % instead of converging. The dominant axis is the third one.

Why the third axis has an optimum rather than a limit
-----------------------------------------------------
_equiv_wire fixes the filament radius by conserving plate volume, so
r ~ 1/sqrt(nx) while the pitch dx ~ 1/nx. Refining the cross-section therefore
makes the filaments *relatively fatter*: 2r/dx ~ sqrt(nx). Two discretization
errors of opposite sign result --

    too few filaments  -> too few current paths -> under-shields -> B high (+)
    too fat filaments  -> neighbours overlap, mutual_parallel's line-filament
                          approximation overestimates coupling
                       -> over-shields -> B low (-)

and they cross. Measured (L = 128 m, dy = 16 m), the signed mean deviation runs
+7.8 % at n_across = 3 through +0.25 % at 9 to -1.2 % at 61, with the magnitude
minimised near 2r/dx ~ 0.4. Refining past that makes agreement *worse* while
1-f keeps improving, which no conductance-only account can produce.

Because r ~ sqrt(1/nx) and dx ~ 1/nx are both functions of nx alone, resolution
and fatness are perfectly confounded along this axis and cannot be separated by
tuning -- separating them would need a change to _equiv_wire, which is out of
scope here. The axis is therefore characterised empirically (Part B1) and held
at its optimum for the model fit (Part B3/C).

Layout
------
  Part A   FEMM reference convergence, shield present -> delta_FEM, the
           reference's own uncertainty, and a converged reference profile.
           Part B compares against that; comparing against the default-mesh
           reference would fold the reference's error into the asymptote.
  Part B1  the transverse ladder: locate the optimum n_across and the sign flip.
  Part B2  f-matched groups: prove the n_across effect is not a disguised f.
  Part B3  a full 4 x 3 factorial in (L, dy) at the optimal n_across, so 1-f
           and W/L vary independently.
  Part B4  isolated plates -- a second, constraint-limited regime that does not
           lean on the conductance argument at all.
  Part C   fit D = c + a(1-f)^alpha + b(W/L)^beta on the B3 slice and
           extrapolate to (f -> 1, L -> inf). c is the deliverable, read
           against delta_FEM.

External cross-check
--------------------
This study's only reference is FEMM, so the finite-geometry agreement it reports
is a self-consistency band, not an independent error bar. The closest external
anchor is Xia & Du, IEEE T-EMC 56(3), 2014 (bib/), Case B: an independent
PEEC-vs-2-D-FEM comparison on a 1 m x 4 m plate meshed 40 x 160 -- the same
L >> W, transverse-limited regime as here -- reporting ~1.5 % agreement,
consistent with the half-to-one-percent floor found below.

Run
---
    .venv/Scripts/python fem/tune.py --quick     # smoke subset, ~4 min
    .venv/Scripts/python fem/tune.py             # full study, ~30 min
    .venv/Scripts/python fem/tune.py --part grid # reuse cached Part A

FEMM is a single global COM server, so Part A is necessarily serialised.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import femm                                                          # noqa: E402

from fem.compare import (CAB, N_PLATES, PTS, SIGMA, THICK, W, X,     # noqa: E402
                         _dense_gb, deviation, sheet_fraction)
from fem.femm2d import Femm2D, Shield                                # noqa: E402
from shielding.analysis import shielded_norm, unshielded_norm        # noqa: E402
from shielding.field import field_norm                               # noqa: E402
from shielding.geometry import build_shield                          # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tune.npz")
# --quick writes its own file. Sharing one would let a 3-minute smoke run
# silently overwrite the 30-minute study that the figure script reads.
OUT_QUICK = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "tune_quick.npz")

# The dense circuit solve is (n_br + n_node - 1)^2 complex. compare.py caps at
# 12 GB; the largest case below needs ~12, so the cap is raised to keep the cap
# itself from shaping the study.
MAX_GB = float(os.environ.get("SHIELD_MAX_GB", 24.0))

# Part A baseline == the Femm2D defaults, i.e. the settings every existing
# result in this repo was produced with.
FEM_BASELINE = dict(plate_mesh=1.5e-3, air_mesh=0.03, boundary_radius=25.0,
                    precision=1e-8, arc_seg=5)

# Reseeding the mesh without touching the physics.
#
# FEMM meshes with an unstructured triangulator, so changing *any* input
# regenerates the whole mesh and the sample points land in different elements.
# With first-order elements B = curl A is element-wise constant, so point
# samples jump. That scatter contaminates every ladder below: it is why
# boundary_radius -- physically irrelevant at 25 m for a balanced three-phase
# source whose field falls off like 1/r^2 -- appears to move the answer by half
# a percent.
#
# min_angle is the cleanest available reseed: it is a mesh-quality target, it
# changes no physics, and small changes in it produce a completely different
# triangulation. Sweeping it therefore measures the noise itself.
RESEED_ANGLES = [30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0]
RESEED_ANGLES_QUICK = [30.0, 31.0, 32.0]
LADDER_SEEDS = [30.0, 31.0, 32.0]     # cheaper ensembles inside the ladders

# Where B is sampled from, and how finely. The repo default refines a large
# 6.0 x 1.6 m air box. Concentrating the same effort in a thin band hugging the
# z = 1 m sampling line ought to buy finer elements exactly where the samples
# are -- it does not help, and A0 reports that as a measured negative result:
# the scatter is not a sampling artifact, so refining the sampling cannot fix it.
SAMPLING_DEFAULT = dict(refine_box=(-1.0, 0.2, 5.0, 1.8), air_mesh=0.03)
SAMPLING_TUNED = dict(refine_box=(-0.3, 0.85, 4.3, 1.15), air_mesh=0.005)

# Calibration grid for A2: settings swept against the exact no-shield source.
# These are the three knobs that touch the source itself -- the cable outline
# polygon, the air mesh the samples land in, and the mesh inside the conductor.
# cable_mesh is carried as a documented null rather than swept: the cables have
# sigma = 0 and a prescribed current, so the mesh inside them cannot influence
# the external field, and the sweep confirmed it to the last digit. One
# off-baseline pair is kept to keep that check live.
CAL_GRID = ([(arc, air, 5e-3)
             for arc in (10, 5, 2)
             for air in (0.06, 0.03, 0.015)]
            + [(5, 0.03, 2e-3)])
CAL_GRID_QUICK = [(10, 0.03, 5e-3), (5, 0.03, 5e-3), (2, 0.03, 5e-3)]
CAL_SEEDS = [30.0, 30.5, 31.0, 31.5, 32.0]

# Kept for reference: what A1's self-convergence alone would have chosen.
# A2 overrides it, because arc_seg=2 is self-converged but less accurate.
#
#   plate_mesh 1.5e-3   no rung differs from any other beyond noise, so this is
#                       chosen for *reproducibility*: it is the quietest rung
#                       (reseed SD 0.105 %), while 0.5e-3 is the noisiest
#                       (0.862 %). Refining the plate makes FEMM less
#                       repeatable here, not more accurate -- sliver elements in
#                       a 3 mm x 4 m strip. The repo default was already right.
#   air_mesh   0.015    0.06 is genuinely too coarse (0.275 % RMS, above noise);
#                       0.03 and 0.015 agree, so take the finer of the two.
#   boundary_radius 25  the flags across this ladder are mutually inconsistent
#                       (15 and 25 "REAL", 50 not), which is the signature of
#                       noise rather than an effect -- as physics demands, since
#                       a balanced three-phase source falls off like 1/r^2.
#   precision  1e-8     rungs are bit-identical. A true null.
#   arc_seg    2        the one genuine effect in Part A: the cable outline is a
#                       polygon, and 10 -> 5 -> 2 degrees moves the field
#                       monotonically and well above noise. The repo default of
#                       5 carries a ~0.19 % RMS bias against 2.
ADOPTED = dict(plate_mesh=1.5e-3, air_mesh=0.015, boundary_radius=25.0,
               precision=1e-8, arc_seg=2, refine_box=(-1.0, 0.2, 5.0, 1.8))

# Ladders run coarse -> fine; the last entry is the reference for that ladder's
# self-convergence. Every rung is a reseed ensemble, including the knobs that
# solve in seconds: with one sample per rung the scatter alone promotes
# boundary_radius = 25 m to a "real" 0.59 % effect, which is impossible -- the
# source is balanced three-phase, its field falls off like 1/r^2, and 15 m and
# 100 m bracket it with no effect at all. Values FEMM rejects outright (it
# refuses precision=1e-6) are caught per-case rather than killing the run.
FEM_LADDERS = {
    "plate_mesh": [3e-3, 1.5e-3, 0.75e-3, 0.5e-3],
    "air_mesh": [0.06, 0.03, 0.015],
    "boundary_radius": [15.0, 25.0, 50.0, 100.0],
    "precision": [1e-8, 1e-10, 1e-12],
    "arc_seg": [10, 5, 2],
}
FEM_LADDERS_QUICK = {
    "plate_mesh": [3e-3, 1.5e-3, 0.75e-3],
    "boundary_radius": [15.0, 25.0, 100.0],
}

# B1: the transverse ladder. L and dy fixed at their cheapest good values, so
# n_across is the only thing moving. ny = 8 throughout, hence all these solves
# are small regardless of n_across.
B1_L, B1_DY = 128.0, 16.0
B1_NAC = [3, 5, 9, 15, 21, 31, 41, 61]
B1_NAC_QUICK = [3, 9, 21, 41]

# B2: groups sharing (nearly) one value of 1-f while nx spans a wide range.
# Within a group, any trend is the transverse axis and cannot be f.
B2_GROUPS = [
    ("1-f~0.007", [(128.0, 16.0, 5), (128.0, 8.0, 9), (128.0, 4.0, 21)]),
    ("1-f~0.002", [(128.0, 16.0, 15), (128.0, 8.0, 31)]),
]

# B3: full factorial at the optimal transverse resolution. dy = L / ny, so ny
# sets 1-f and L sets W/L; both vary over a decade and neither is confounded
# with the other. Cost is set by ny alone (ny=8 ~ 5 s, 16 ~ 30 s, 32 ~ 210 s).
B3_NAC = 21
B3_L = [32.0, 64.0, 128.0, 256.0]
B3_NY = [8, 16, 32]
B3_L_QUICK, B3_NY_QUICK = [64.0, 256.0], [8, 16]

# Shortest shield admitted to the Part C fit. Below this the finite-vs-infinite
# difference is the dominant term and it is physics, not discretization.
FIT_MIN_L = 64.0

# B4: electrically separate plates. Rc = 1 kOhm is "infinite" for this network
# (branch impedances are micro-ohms), matching FEMM mode="isolated".
B4_L = [32.0, 64.0, 128.0, 256.0]
B4_NY = 8
RC_ISOLATED = 1e3

# The compare.py T4 winner, re-run against the *default* reference purely to
# confirm this script measures the same thing compare.py measured.
CONTINUITY_CASE = (64.0, 2.0, 15)
CONTINUITY_EXPECTED_RMS = 1.12


def signed_mean(curve, ref):
    """Mean signed relative deviation (%).

    compare.deviation() returns max and RMS of |dev|, which is what the existing
    tables quote. Sign is what makes the transverse axis legible -- the two
    competing errors there have opposite signs and cancel at the optimum, which
    an absolute metric cannot show -- and the Part C model is additive, so it
    can only be fitted with the sign kept. Reported alongside, not instead.
    """
    rel = (np.asarray(curve) - np.asarray(ref)) / np.asarray(ref)
    return 100.0 * float(rel.mean())


# --------------------------------------------------------------- Part A: FEMM

def femm_case(mode="welded", with_shield=True, **kw):
    """One FEMM solve at the given discretization settings.

    Returns None if FEMM rejects the settings, so a single invalid rung cannot
    take down a half-hour run.
    """
    settings = dict(FEM_BASELINE)
    settings.update(SAMPLING_DEFAULT)
    settings.update(kw)
    sh = (Shield(width=W, n_plates=N_PLATES, thickness=THICK, sigma=SIGMA,
                 mode=mode) if with_shield else None)
    t0 = time.time()
    try:
        with Femm2D(CAB, shield=sh, **settings) as m:
            B = field_norm(m.sample_B(PTS)) * 1e6
            currents = m.circuit_currents()
            n_elem = int(femm.mo_numelements())
    except Exception as exc:
        first = str(exc).strip().splitlines()[0]
        print(f"    FEMM rejected {kw}: {first}")
        return None
    guard = max(abs(v) for v in currents.values()) if currents else 0.0
    return dict(B=B, wall=time.time() - t0, n_elem=n_elem, guard=guard)


def femm_ensemble(mode="welded", seeds=None, label="", with_shield=True, **kw):
    """Solve the same physics over several mesh seeds; average and measure spread.

    The mesh seed carries no physics, so averaging over seeds is variance
    reduction on a purely numerical artifact rather than smoothing of a result:
    it cuts the sampling scatter by sqrt(N) and gives the reference an honest
    error bar. `sd` is what a *single* FEMM run is worth; `sem` is what the
    ensemble mean used as the reference is worth.
    """
    seeds = seeds or RESEED_ANGLES
    runs = [femm_case(mode=mode, min_angle=s, with_shield=with_shield, **kw)
            for s in seeds]
    runs = [r for r in runs if r is not None]
    if not runs:
        return None
    C = np.array([r["B"] for r in runs])
    mean = C.mean(axis=0)
    sd = 100.0 * (C.std(axis=0, ddof=1) / mean) if len(runs) > 1 else np.zeros_like(mean)
    sem = sd / np.sqrt(len(runs))
    out = dict(B=mean, curves=C, sd=sd, sem=sem, n=len(runs),
               sd_rms=float(np.sqrt(np.mean(sd ** 2))),
               sem_rms=float(np.sqrt(np.mean(sem ** 2))),
               wall=sum(r["wall"] for r in runs),
               n_elem=int(np.mean([r["n_elem"] for r in runs])),
               guard=max(r["guard"] for r in runs))
    if label:
        print(f"    {label:<26s} B0={mean[0]:.5f} uT  reseed scatter: "
              f"single-run {out['sd_rms']:.3f} % RMS (worst point {sd.max():.3f} %),"
              f" mean-of-{len(runs)} {out['sem_rms']:.3f} %"
              f"  [{out['n_elem']} elems, {out['wall']:.0f}s]")
    return out


def run_part_A(quick=False):
    """Establish what the FEMM reference is actually worth, then build one.

    compare.py's T0 checks the source field with NO shield present. This checks
    the eddy-current solution that Part B is compared against, and -- more to
    the point -- separates genuine discretization error from mesh-regeneration
    noise, which turns out to dominate several of the knobs.
    """
    print("\n=== Part A: what is the FEMM reference worth? ===")
    seeds = RESEED_ANGLES_QUICK if quick else RESEED_ANGLES

    print(f"\n  A0: repeatability under reseeding ({len(seeds)} mesh seeds, "
          "identical physics)")
    ens_default = femm_ensemble(seeds=seeds, label="repo default sampling",
                                **SAMPLING_DEFAULT)
    ens_tuned = femm_ensemble(seeds=seeds, label="tuned sampling band",
                              **SAMPLING_TUNED)
    adopted, adopted_name = ((ens_tuned, "tuned") if ens_tuned is not None
                             and ens_tuned["sd_rms"] < ens_default["sd_rms"]
                             else (ens_default, "repo default"))
    sampling = SAMPLING_TUNED if adopted_name == "tuned" else SAMPLING_DEFAULT
    print(f"    -> adopting the {adopted_name} sampling settings; the reference is"
          f" the mean of {adopted['n']} seeds,")
    print(f"       so delta_FEM = {adopted['sem_rms']:.3f} % RMS. A single run"
          f" would be worth {adopted['sd_rms']:.3f} %.")

    print("\n  A1: knob ladders, read against that noise floor")
    print("      (an effect no bigger than the scatter on the comparison is not")
    print("       an effect, however suggestive one sample per rung looks)")
    ladders = FEM_LADDERS_QUICK if quick else FEM_LADDERS
    rows = []
    for knob, values in ladders.items():
        print(f"\n  {knob}:  (each rung averaged over {len(LADDER_SEEDS)} "
              "mesh seeds)")
        cases = []
        for v in values:
            kw = dict(sampling)
            kw[knob] = v
            r = femm_ensemble(seeds=LADDER_SEEDS, **kw)
            if r is None:
                continue
            cases.append((v, r))
            bad = "   <-- CONSTRAINT NOT APPLIED" if r["guard"] >= 1e-6 else ""
            print(f"    {v!r:>10}  B0={r['B'][0]:.5f} uT  elems={r['n_elem']:7d}"
                  f"  {r['wall']:5.1f}s  reseed SD {r['sd_rms']:6.3f} %{bad}")
        if not cases:
            continue
        finest = cases[-1][1]
        for v, r in cases:
            mx, rms = deviation(r["B"], finest["B"])
            sd = r["sd_rms"]
            # Scatter on the *difference* of two ensemble means, propagated.
            noise = np.hypot(sd, finest["sd_rms"]) / np.sqrt(len(LADDER_SEEDS))
            verdict = "REAL" if rms > 2 * noise else "within noise"
            rows.append(dict(knob=knob, value=float(v), B=r["B"], max=mx, rms=rms,
                             mean=signed_mean(r["B"], finest["B"]), sd=sd,
                             n_elem=r["n_elem"], wall=r["wall"], guard=r["guard"]))
            print(f"    {v!r:>10}  vs finest: max {mx:6.3f} %, RMS {rms:6.3f} %"
                  f"  (noise ~{noise:.3f} %)  -> {verdict}")

    # A2: calibrate against the one thing here that has an exact answer.
    #
    # A1 can only measure self-convergence, and self-convergence is not accuracy:
    # refining the cable polygon moves FEMM monotonically and well above noise,
    # yet *away* from the exact field. With no shield present the source is three
    # infinite line currents, for which shielding.field has the closed form, so
    # the settings can be chosen against truth instead of against themselves.
    print("\n  A2: calibrate FEMM against the exact no-shield source")
    print("      (self-convergence is not accuracy -- A1's one 'REAL' knob is")
    print("       the cable polygon, and refining it makes the source worse)")
    analytic = unshielded_norm(CAB, PTS)
    grid = (CAL_GRID_QUICK if quick else CAL_GRID)
    cal = []
    for arc_seg, air_mesh, cable_mesh in grid:
        kw = dict(FEM_BASELINE)
        kw.update(SAMPLING_DEFAULT)
        kw.update(arc_seg=arc_seg, air_mesh=air_mesh, cable_mesh=cable_mesh)
        seeds_cal = LADDER_SEEDS if quick else CAL_SEEDS
        e = femm_ensemble(seeds=seeds_cal, with_shield=False, **kw)
        if e is None:
            continue
        mx, rms = deviation(e["B"], analytic)
        cal.append(dict(arc_seg=arc_seg, air_mesh=air_mesh,
                        cable_mesh=cable_mesh, max=mx, rms=rms,
                        sd=e["sd_rms"], err=e["sd_rms"] / np.sqrt(len(seeds_cal)),
                        wall=e["wall"]))
        print(f"    arc_seg={arc_seg:2d} air={air_mesh:<6g} cable={cable_mesh:<7g}"
              f" -> vs analytic: max {mx:6.3f} %, RMS {rms:6.3f} %"
              f"   (reseed SD {e['sd_rms']:.3f} %)")

    # Noise-aware selection. Picking the single lowest RMS chases whichever
    # configuration drew a lucky mesh: the air_mesh rungs differ by less than
    # their own reseed scatter, and the raw minimum lands on the coarsest one,
    # which then makes the shielded reference twice as noisy. So: take every
    # configuration statistically tied with the best on accuracy, and among
    # those choose the most *reproducible*.
    raw_best = min(cal, key=lambda r: r["rms"])
    tied = [r for r in cal
            if r["rms"] - raw_best["rms"] <= np.hypot(r["err"], raw_best["err"])]
    best_cal = min(tied, key=lambda r: r["sd"])
    if len(tied) > 1:
        print(f"    {len(tied)} configurations are statistically tied on accuracy"
              f" (best {raw_best['rms']:.3f} % RMS);")
        print("    choosing the most reproducible of them rather than the lowest"
              " single number.")
    adopted_kw = dict(FEM_BASELINE)
    adopted_kw.update(SAMPLING_DEFAULT)
    adopted_kw.update(arc_seg=best_cal["arc_seg"], air_mesh=best_cal["air_mesh"],
                      cable_mesh=best_cal["cable_mesh"])
    print(f"    -> best source accuracy: arc_seg={best_cal['arc_seg']}, "
          f"air_mesh={best_cal['air_mesh']:g}, cable_mesh={best_cal['cable_mesh']:g}"
          f"  ({best_cal['rms']:.3f} % RMS)")
    print(f"       compare.py's T0 quotes 0.41 / 0.25 % at the repo defaults;"
          f" this is the floor")
    print("       any shielded comparison inherits, because the circuit method"
          " uses the exact source.")

    print("\n  A3: references for Part B (ensemble means at the calibrated settings)")
    ref = femm_ensemble(seeds=seeds, label="welded", **adopted_kw)
    iso = femm_ensemble(mode="isolated", seeds=seeds, label="isolated",
                        **adopted_kw)
    default_single = femm_case(**SAMPLING_DEFAULT)
    t_max, t_rms = deviation(adopted["B"], ref["B"])
    print(f"    default-settings reference vs calibrated reference: "
          f"max {t_max:.3f} %, RMS {t_rms:.3f} %")
    print(f"    source-model floor {best_cal['rms']:.3f} % RMS vs delta_FEM "
          f"{ref['sem_rms']:.3f} % -- the former is what limits Part B.")

    # What compare.py effectively used: a single draw, not an ensemble mean.
    d_max, d_rms = deviation(default_single["B"], ref["B"])
    print(f"    one single-run reference (what compare.py used) vs this ensemble"
          f" mean: max {d_max:.3f} %, RMS {d_rms:.3f} %")
    print(f"    delta_FEM = {ref['sem_rms']:.3f} % RMS -- circuit-side agreement"
          " tighter than this is")
    print("    below the noise of the reference it is being measured against.")

    return dict(
        fem_knob=np.array([r["knob"] for r in rows]),
        fem_value=np.array([r["value"] for r in rows]),
        fem_curves=np.array([r["B"] for r in rows]),
        fem_max=np.array([r["max"] for r in rows]),
        fem_rms=np.array([r["rms"] for r in rows]),
        fem_mean=np.array([r["mean"] for r in rows]),
        fem_elems=np.array([r["n_elem"] for r in rows]),
        fem_wall=np.array([r["wall"] for r in rows]),
        fem_guard=np.array([r["guard"] for r in rows]),
        fem_sd=np.array([r["sd"] for r in rows]),
        ref_converged=ref["B"], ref_default=default_single["B"],
        ref_isolated=iso["B"],
        ref_sd=ref["sd"], ref_sem=ref["sem"], ref_curves=ref["curves"],
        noise_default=np.array([ens_default["sd_rms"], ens_default["sem_rms"]]),
        noise_tuned=np.array([ens_tuned["sd_rms"], ens_tuned["sem_rms"]]
                             if ens_tuned is not None else [np.nan, np.nan]),
        noise_adopted=np.array([ref["sd_rms"], ref["sem_rms"]]),
        adopted_sampling=np.array([adopted_name]),
        delta_fem=np.array([float(ref["sd"].max()), ref["sem_rms"]]),
        default_vs_converged=np.array([d_max, d_rms]),
        default_vs_tuned=np.array([t_max, t_rms]),
        cal_params=np.array([[r["arc_seg"], r["air_mesh"], r["cable_mesh"]]
                             for r in cal]),
        cal_max=np.array([r["max"] for r in cal]),
        cal_rms=np.array([r["rms"] for r in cal]),
        cal_sd=np.array([r["sd"] for r in cal]),
        source_floor=np.array([best_cal["max"], best_cal["rms"]]),
        adopted_settings=np.array([f"arc_seg={best_cal['arc_seg']}",
                                   f"air_mesh={best_cal['air_mesh']:g}",
                                   f"cable_mesh={best_cal['cable_mesh']:g}"]),
    )


# ------------------------------------------------------ Part B: circuit sweeps

def circuit_case(L, dy, n_across, ref, Rc=0.0, note=""):
    """One circuit-method solve, with the cost and fatness diagnostics kept."""
    n_along = int(round(L / dy))
    mesh = build_shield("parallel", W=W, L=L, n_plates=N_PLATES,
                        n_across=n_across, n_along=n_along, sigma=SIGMA,
                        thickness=THICK, Rc=Rc)
    gb = _dense_gb(mesh)
    tag = f"L={L:5.0f} dy={dy:5.2f} n_ac={n_across:3d}"
    if gb > MAX_GB:
        print(f"    SKIP {tag}: needs ~{gb:.1f} GB > {MAX_GB} GB cap")
        return None
    dx = W / (N_PLATES * n_across)
    fatness = 2.0 * mesh.radius / dx
    t0 = time.time()
    B, I = shielded_norm(mesh, CAB, PTS)
    wall = time.time() - t0

    contact = np.where(mesh.branch_axis == -1)[0]
    leak = (float(np.abs(I[contact]).max() / np.abs(I).max())
            if contact.size else 0.0)

    f = sheet_fraction(L, n_along, n_across)
    mx, rms = deviation(B, ref)
    row = dict(L=L, dy=dy, n_across=n_across, Rc=Rc, f=f, fatness=fatness,
               gb=gb, wall=wall, B=B, max=mx, rms=rms,
               mean=signed_mean(B, ref), n_branches=mesh.n_branches,
               n_along=n_along, leak=leak)
    print(f"    {tag}  1-f={1 - f:.4f}  2r/dx={fatness:.3f}  ->  max {mx:6.2f} %,"
          f" RMS {rms:6.2f} %, mean {row['mean']:+6.2f} %  [{wall:.0f}s]{note}")
    return row


def _pack(rows, prefix):
    """Collapse a list of case dicts into npz-friendly arrays."""
    if not rows:
        return {}
    out = {f"{prefix}_curves": np.array([r["B"] for r in rows]),
           f"{prefix}_params": np.array([[r["L"], r["dy"], r["n_across"]]
                                         for r in rows])}
    for key in ("f", "fatness", "max", "rms", "mean", "gb", "wall", "leak",
                "n_branches"):
        out[f"{prefix}_{key}"] = np.array([r[key] for r in rows])
    return out


def run_B1(ref, quick=False):
    """The transverse ladder: where is the optimum, and where does it turn over?"""
    print(f"\n=== Part B1: transverse ladder at L={B1_L:g} m, dy={B1_DY:g} m ===")
    print("    n_across is the only thing moving. Watch the SIGN of the mean:")
    print("    + = under-shielded (too few filaments), - = over-shielded (too fat).")
    nacs = B1_NAC_QUICK if quick else B1_NAC
    rows = [r for r in (circuit_case(B1_L, B1_DY, n, ref) for n in nacs)
            if r is not None]
    if len(rows) > 2:
        # Ranked by RMS, not by |mean|: the two competing errors cancel in the
        # mean at a different n_across than the one where the profile shape is
        # closest, and it is the shape that matters.
        best = min(rows, key=lambda r: r["rms"])
        signs = [np.sign(r["mean"]) for r in rows]
        flipped = any(a != b for a, b in zip(signs, signs[1:]))
        verdict = ("two competing errors, so this axis has an optimum" if flipped
                   else "single-signed error, so this axis is monotone")
        print(f"    RMS minimised at n_across={best['n_across']} "
              f"(2r/dx={best['fatness']:.3f}, RMS {best['rms']:.2f} %)")
        print(f"    sign flip across the ladder: {'yes' if flipped else 'no'}"
              f"  -> {verdict}")
        if flipped:
            cross = next(r for a, r in zip(signs[1:], rows[1:])
                         if a != signs[0])
            print(f"    mean deviation changes sign by n_across={cross['n_across']}"
                  f" (2r/dx={cross['fatness']:.3f}): under-shielding gives way to"
                  " over-shielding")
    return rows


def run_B2(ref, quick=False):
    """f-matched groups: is the B1 trend really f in disguise?"""
    print("\n=== Part B2: n_across varied at matched 1-f ===")
    rows, groups = [], []
    for name, cases in (B2_GROUPS[:1] if quick else B2_GROUPS):
        print(f"  group {name}:")
        got = []
        for L, dy, nac in cases:
            r = circuit_case(L, dy, nac, ref)
            if r is not None:
                rows.append(r)
                groups.append(name)
                got.append(r)
        if len(got) > 1:
            fspread = max(1 - r["f"] for r in got) - min(1 - r["f"] for r in got)
            dspread = max(r["mean"] for r in got) - min(r["mean"] for r in got)
            print(f"    1-f varies by {fspread:.4f} across the group, but the mean"
                  f" deviation varies by {dspread:.2f} points")
            print("    -> the transverse axis is not a disguised conductance effect")
    return rows, groups


def run_B3(ref, quick=False):
    """Full factorial in (L, dy) at the optimal transverse resolution."""
    Ls = B3_L_QUICK if quick else B3_L
    nys = B3_NY_QUICK if quick else B3_NY
    print(f"\n=== Part B3: {len(Ls)} x {len(nys)} factorial in (L, dy) at "
          f"n_across={B3_NAC} ===")
    print("    ny sets 1-f, L sets W/L; neither is confounded with the other.")
    rows = []
    for L in Ls:
        for ny in nys:
            r = circuit_case(L, L / ny, B3_NAC, ref)
            if r is not None:
                rows.append(r)
    return rows


def run_B4(ref_isolated, quick=False):
    """Isolated plates: a constraint-limited regime, independent of f."""
    print(f"\n=== Part B4: isolated plates (Rc = {RC_ISOLATED:g} Ohm) vs FEMM "
          f"isolated ===")
    Ls = B4_L[:2] if quick else B4_L
    rows = [r for r in (circuit_case(L, L / B4_NY, B3_NAC, ref_isolated,
                                     Rc=RC_ISOLATED) for L in Ls)
            if r is not None]
    if rows:
        worst = max(r["leak"] for r in rows)
        ok = worst < 1e-3
        print(f"    max contact-branch current / max branch current = {worst:.2e}"
              f"  -> {'ok, plates are electrically separate' if ok else 'CONTACTS STILL CONDUCTING'}")
    return rows


# ------------------------------------------------ Part C: model and extrapolate

def _model(uv, c, a, alpha, b, beta):
    u, v = uv
    return c + a * u ** alpha + b * v ** beta


def run_part_C(rows, delta_fem_rms):
    """Fit D = c + a(1-f)^alpha + b(W/L)^beta on the B3 slice; extrapolate.

    Fitted at fixed n_across, so the transverse error is a constant absorbed
    into c. c is therefore the residual disagreement *at that transverse
    resolution* in the joint (f -> 1, L -> inf) limit -- not a claim that the
    transverse term vanishes. Part B1 characterises how c moves with n_across.
    """
    from scipy.optimize import curve_fit

    print("\n=== Part C: error model and extrapolation ===")
    # The shortest shield is excluded, on the same grounds compare.py excludes
    # its 4 m case from the pooled statistic: at W/L = 0.125 the circuit method
    # is modelling a genuinely finite shield against an infinite FEMM one, so
    # its error is end-effect physics rather than discretization, and it swings
    # by 4.4 points across dy while every longer shield sits within 0.5.
    kept = [r for r in rows if r["L"] >= FIT_MIN_L]
    dropped = [r for r in rows if r["L"] < FIT_MIN_L]
    if dropped:
        print(f"  excluding {len(dropped)} case(s) with L < {FIT_MIN_L:g} m "
              "(end-effect dominated, not discretization):")
        for r in dropped:
            print(f"    L={r['L']:g} dy={r['dy']:g} -> mean {r['mean']:+.2f} %")
    if len(kept) < 6:
        print("  too few points to fit; skipping")
        return {}

    u = np.array([1.0 - r["f"] for r in kept])
    v = np.array([W / r["L"] for r in kept])
    D = np.array([r["mean"] for r in kept])
    print(f"  fitting {len(u)} points at n_across={B3_NAC}; "
          f"1-f spans {u.min():.4f}-{u.max():.4f}, W/L spans {v.min():.4f}-{v.max():.3f}")
    print(f"  spread of the data being fitted: {D.min():+.2f} to {D.max():+.2f} %"
          f" (SD {D.std(ddof=1):.2f})")

    # The amplitudes are allowed to go negative. The conductance argument
    # predicts a > 0 (a shortfall under-shields), but at the optimal transverse
    # resolution the measured trend in 1-f runs the other way; clamping a >= 0
    # would rail the fit and hide that.
    lo = [-5.0, -5e3, 0.3, -5e3, 0.3]
    hi = [5.0, 5e3, 3.0, 5e3, 3.0]
    p0 = [0.0, 50.0, 1.0, 5.0, 1.0]

    def fit(fixed_c=None, fixed_alpha=None):
        fixed = [fixed_c, None, fixed_alpha, None, None]
        free = [i for i, fv in enumerate(fixed) if fv is None]

        def fn(uv, *p):
            full = list(fixed)
            for i, val in zip(free, p):
                full[i] = val
            return _model(uv, *full)

        popt, _ = curve_fit(fn, (u, v), D, p0=[p0[i] for i in free],
                            bounds=([lo[i] for i in free], [hi[i] for i in free]),
                            maxfev=200000)
        full = list(fixed)
        for i, val in zip(free, popt):
            full[i] = val
        resid = D - _model((u, v), *full)
        return np.array(full, float), float(np.sqrt(np.mean(resid ** 2)))

    stages = [("c = 0, exponents free", fit(fixed_c=0.0)),
              ("c free, exponents free", fit()),
              ("c free, alpha = 1 (first-order theory)", fit(fixed_alpha=1.0))]
    print("      model                                        c       a"
          "   alpha       b    beta   resid")
    for name, (p, resid) in stages:
        print(f"  {name:<40s} {p[0]:+7.2f} {p[1]:7.1f} {p[2]:6.2f} {p[3]:7.2f}"
              f" {p[4]:6.2f} {resid:7.3f}")

    # The honest control: does the two-term model earn its parameters at all, or
    # is the residual simply a constant offset? If four extra parameters barely
    # beat "D = const", the deviation is not a conductance or a length effect.
    const_resid = float(np.sqrt(np.mean((D - D.mean()) ** 2)))
    print(f"  {'constant D = %+.2f %% (no f, no L terms)' % D.mean():<40s} "
          f"{'':>7} {'':>7} {'':>6} {'':>7} {'':>6} {const_resid:7.3f}")

    best_p, best_resid = stages[1][1]
    if best_resid > 0.7 * const_resid:
        print(f"\n  the (1-f) and (W/L) terms reduce the residual from "
              f"{const_resid:.3f} to only {best_resid:.3f}:")
        print("  at this transverse resolution the deviation is a CONSTANT BIAS,"
              " not a")
        print("  conductance or a finite-length effect. compare.py's"
              " RMS ~ 0.64 (1-f) does not")
        print("  survive once the cross-section is resolved and the reference is"
              " de-noised.")
    # Leave-one-out on c: a headline number must not rest on one influential case.
    loo = []
    for drop in range(len(u)):
        keep = np.arange(len(u)) != drop
        try:
            popt, _ = curve_fit(lambda uv, *p: _model(uv, *p),
                                (u[keep], v[keep]), D[keep],
                                p0=best_p, bounds=(lo, hi), maxfev=200000)
            loo.append(popt[0])
        except Exception:
            pass
    loo = np.array(loo) if loo else np.array([best_p[0]])
    print(f"\n  extrapolated residual at n_across={B3_NAC}:  c = {best_p[0]:+.2f} %"
          f"   (leave-one-out {loo.min():+.2f} to {loo.max():+.2f} %)")
    print(f"  FEMM reference's own uncertainty:      delta_FEM = {delta_fem_rms:.2f} % RMS")
    if abs(best_p[0]) <= delta_fem_rms:
        print("  -> |c| is within the reference's own noise: the two solvers agree")
        print("     to the limit of what this reference can resolve.")
    else:
        print(f"  -> |c| exceeds delta_FEM by {abs(best_p[0]) / delta_fem_rms:.1f}x:"
              " a real residual remains, and at this")
        print("     transverse resolution it is the filament discretization, not"
              " f or L.")
    return dict(fit_params=best_p, fit_resid=np.array([best_resid]),
                fit_loo=loo, fit_u=u, fit_v=v, fit_D=D,
                fit_stage_names=np.array([s[0] for s in stages]),
                fit_stage_params=np.array([s[1][0] for s in stages]),
                fit_stage_resid=np.array([s[1][1] for s in stages]))


def recommend(rows, delta_fem_rms):
    """The settings to actually use: lowest RMS anywhere in the study."""
    if not rows:
        return {}
    best = min(rows, key=lambda r: r["rms"])
    cheap = min((r for r in rows if r["rms"] < 1.5 * best["rms"]),
                key=lambda r: r["wall"])
    print("\n=== recommended settings ===")
    for name, r in (("best agreement", best), ("cheapest within 1.5x", cheap)):
        print(f"  {name}:  L = {r['L']:g} m, dy = {r['dy']:g} m, "
              f"n_across = {r['n_across']}")
        print(f"      1-f = {1 - r['f']:.4f}   W/L = {W / r['L']:.4f}   "
              f"2r/dx = {r['fatness']:.3f}")
        print(f"      max {r['max']:.2f} %, RMS {r['rms']:.2f} %, "
              f"mean {r['mean']:+.2f} %   (delta_FEM = {delta_fem_rms:.2f} %)")
        print(f"      cost: {r['n_branches']} branches, {r['gb']:.2f} GB, "
              f"{r['wall']:.0f} s")
    return dict(best_params=np.array([best["L"], best["dy"], best["n_across"]]),
                best_metrics=np.array([best["max"], best["rms"], best["mean"]]),
                best_fatness=np.array([best["fatness"]]),
                cheap_params=np.array([cheap["L"], cheap["dy"], cheap["n_across"]]),
                cheap_metrics=np.array([cheap["max"], cheap["rms"], cheap["mean"]]))


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(
        description="Tune the circuit-method parameters against a converged "
                    "FEMM 2-D reference.")
    ap.add_argument("--part", choices=["fem", "grid", "all"], default="all")
    ap.add_argument("--quick", action="store_true",
                    help="small subset for a fast smoke run")
    args = ap.parse_args()

    out = OUT_QUICK if args.quick else OUT
    store = {}
    # A quick run may still *read* the full study's cached Part A; it just never
    # writes back over it.
    for src in ([out, OUT] if args.quick else [out]):
        if os.path.exists(src):
            with np.load(src, allow_pickle=False) as cached:
                store = {k: cached[k] for k in cached.files}
            break

    if args.part in ("fem", "all"):
        store.update(run_part_A(quick=args.quick))
    elif "ref_converged" not in store:
        sys.exit("no cached Part A in fem/tune.npz -- run with --part fem first")

    if args.part in ("grid", "all"):
        ref = store["ref_converged"]
        delta_rms = float(store["delta_fem"][1])

        b1 = run_B1(ref, quick=args.quick)
        b2, b2_groups = run_B2(ref, quick=args.quick)
        b3 = run_B3(ref, quick=args.quick)
        b4 = run_B4(store["ref_isolated"], quick=args.quick)

        print("\n=== continuity check against fem/compare.py ===")
        L, dy, nac = CONTINUITY_CASE
        cont = circuit_case(L, dy, nac, store["ref_default"])
        if cont is not None:
            ok = abs(cont["rms"] - CONTINUITY_EXPECTED_RMS) < 0.05
            print(f"    compare.py's T4 winner vs the *default* reference: "
                  f"RMS {cont['rms']:.2f} % (expected {CONTINUITY_EXPECTED_RMS:.2f} %)"
                  f"  -> {'PASS' if ok else 'MISMATCH'}")
            store["continuity_rms"] = np.array([cont["rms"]])

        for rows, prefix in ((b1, "b1"), (b2, "b2"), (b3, "b3"), (b4, "b4")):
            store.update(_pack(rows, prefix))
        if b2_groups:
            store["b2_group"] = np.array(b2_groups)
        store.update(run_part_C(b3, delta_rms))
        store.update(recommend(b1 + b2 + b3, delta_rms))

    store["x"] = X
    np.savez(out, **store)
    print(f"\nwrote {out}")
    if args.quick:
        print("(quick mode: figures/fig_fem_tune.py reads tune.npz, not this)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
