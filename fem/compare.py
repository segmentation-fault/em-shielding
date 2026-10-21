"""T0-T3: cross-validate the filament circuit method against FEMM 2-D.

Run this once; it performs every FEMM and circuit-method solve and writes
fem/results.npz, which the figure scripts then only plot. FEMM is a single
global COM server, so the solves are necessarily serialised.

    .venv/Scripts/python fem/compare.py

The four tests, and what each one can actually prove:

  T0  Source model, no shield. FEMM vs the analytic infinite-line field.
      A gate: if this fails nothing below is meaningful.

  T1  FEMM 2-D (infinite shield) vs the circuit method on the real 4 m x 4 m
      shield. These *should* disagree -- 2-D over-predicts shielding because it
      gives the shield infinite length. Quantifies the papers' own criticism.

  T2  The length bridge, and the only quantitative validation here: sweep the
      circuit method's plate length L with the cross-section held fixed and
      sample at mid-length. As L grows this must converge onto the FEMM curve.

  T3  Refine the cross-section at fixed L, and test a specific prediction while
      doing it. _equiv_wire splits the plate metal between x- and y-directed
      filaments by conserving volume, so the mesh's effective sheet conductance
      along the current direction (y, parallel to the cables) is only

          f = (nx+1) L / [ (nx+1) L + (ny+1) W ]

      times the true sigma*t -- roughly 0.6 at Sergeant's standard n_across = 3.
      The model therefore under-shields, and refining n_across improves the
      answer mainly by driving f towards 1 rather than by resolving any field
      detail. If measured deviation tracks (1 - f), the residual gap is the
      resistance model, not the discretization.

  T4  The corrected bridge. Note that with the along-cable spacing dy held
      fixed, ny+1 ~ L/dy, so f -> (nx+1)/[(nx+1) + W/dy] -- INDEPENDENT of L.
      That is why T2 stalls instead of converging: no amount of length can
      remove an L-independent conductance error. T4 therefore drives f towards 1
      (coarser dy, finer cross-section) *while* keeping L large, which is the
      only configuration in which the circuit method and FEMM are solving the
      same problem. This is where the real agreement number comes from.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem.femm2d import Femm2D, Shield                              # noqa: E402
from shielding.analysis import (profile_points, shielded_norm,     # noqa: E402
                                unshielded_norm)
from shielding.field import field_norm                             # noqa: E402
from shielding.geometry import Cables, build_shield                # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.npz")

# Sergeant Table I
SPACING, DEPTH, CURRENT = 0.25, 1.5, 900.0
W, N_PLATES, THICK, SIGMA = 4.0, 8, 3e-3, 5.8e7
DY = 0.25                    # filament spacing along the cables, held fixed

# Skip any circuit solve whose dense system would exceed this. The solves are
# fast (a 4784-unknown system takes ~6 s) but memory grows as n^2, so this is
# the binding constraint. Override with SHIELD_MAX_GB for a smaller machine.
MAX_GB = float(os.environ.get("SHIELD_MAX_GB", 12.0))

X = np.linspace(0.0, 4.0, 21)
PTS = profile_points(X, y=0.0, z=1.0)
CAB = Cables(spacing=SPACING, depth=DEPTH, current=CURRENT)


def deviation(a, b):
    """max and RMS relative deviation of curve `a` from reference `b` (%)."""
    rel = np.abs(np.asarray(a) - np.asarray(b)) / np.asarray(b)
    return 100.0 * rel.max(), 100.0 * float(np.sqrt(np.mean(rel ** 2)))


def _dense_gb(mesh):
    """Peak memory of solve_currents: the (n_br + n_node - 1)^2 complex block
    system, plus the branch impedance matrix and a copy made during the solve."""
    n = mesh.n_branches + mesh.n_nodes - 1
    return (2 * n ** 2 + 2 * mesh.n_branches ** 2) * 16 / 1024 ** 3


def sheet_fraction(L, n_along, n_across):
    """Fraction of the true sheet conductance sigma*t that the filament mesh
    actually presents to current flowing along y.

    _equiv_wire conserves the plate's total metal volume across *both* filament
    directions, but in this geometry the current is essentially unidirectional,
    so the metal allocated to x-filaments is largely idle. This returns the
    resulting shortfall factor -- see the module docstring.
    """
    nx, ny = N_PLATES * n_across, n_along
    return (nx + 1) * L / ((nx + 1) * L + (ny + 1) * W)


def circuit_case(L, n_along, n_across=3, Rc=0.0, config="parallel", label=""):
    """One circuit-method solve, skipped with a warning if it would not fit."""
    mesh = build_shield(config, W=W, L=L, n_plates=N_PLATES, n_across=n_across,
                        n_along=n_along, sigma=SIGMA, thickness=THICK, Rc=Rc)
    gb = _dense_gb(mesh)
    tag = label or f"L={L} n_along={n_along} n_across={n_across} Rc={Rc}"
    if gb > MAX_GB:
        print(f"  SKIP {tag}: dense system needs ~{gb:.1f} GB > {MAX_GB} GB cap")
        return None
    f = sheet_fraction(L, n_along, n_across)
    print(f"  solve {tag}  ({mesh.n_branches} branches, ~{gb:.2f} GB, "
          f"sheet conductance f={f:.3f})")
    B, _ = shielded_norm(mesh, CAB, PTS)
    return B


# --------------------------------------------------------------------- tests

def run_T0():
    print("\n=== T0: source model, no shield ===")
    analytic = unshielded_norm(CAB, PTS)
    with Femm2D(CAB, shield=None) as m:
        femm_uT = field_norm(m.sample_B(PTS)) * 1e6
    mx, rms = deviation(femm_uT, analytic)
    print(f"  analytic B0(0,0,1) = {analytic[0]:.3f} uT")
    print(f"  FEMM     B0(0,0,1) = {femm_uT[0]:.3f} uT   (papers plot ~14 uT)")
    print(f"  max dev {mx:.3f} %, RMS dev {rms:.3f} %"
          f"   -> {'PASS' if mx < 0.5 else 'FAIL'}")
    return analytic, femm_uT, mx


def run_T1():
    print("\n=== T1: FEMM 2-D (infinite shield) vs circuit method (4 m x 4 m) ===")
    out = {}
    for mode in ("welded", "isolated"):
        sh = Shield(width=W, n_plates=N_PLATES, thickness=THICK,
                    sigma=SIGMA, mode=mode)
        with Femm2D(CAB, shield=sh) as m:
            out[f"femm_{mode}"] = field_norm(m.sample_B(PTS)) * 1e6
            cur = m.circuit_currents()
        worst = max(abs(v) for v in cur.values())
        print(f"  FEMM {mode:9s}: max |net circuit current| = {worst:.2e} A"
              f"   -> {'ok' if worst < 1e-6 else 'CONSTRAINT NOT APPLIED'}")

    n_along = int(round(4.0 / DY))
    out["circuit_rc0"] = circuit_case(4.0, n_along, Rc=0.0, label="4x4 Rc=0")
    out["circuit_rc100"] = circuit_case(4.0, n_along, Rc=100e-3, label="4x4 Rc=100mOhm")

    for femm_key, circ_key in (("femm_welded", "circuit_rc0"),
                               ("femm_isolated", "circuit_rc100")):
        if out.get(circ_key) is None:
            continue
        mx, rms = deviation(out[circ_key], out[femm_key])
        print(f"  {circ_key} vs {femm_key}: max {mx:.1f} %, RMS {rms:.1f} %")
    print("  (a large gap here is the expected physics, not an error: the 4 m"
          "\n   shield is finite, the FEMM one is infinitely long)")
    return out


def run_T2(femm_welded):
    print("\n=== T2: length bridge at fixed mesh, circuit method vs FEMM ===")
    lengths, curves, devs, fracs = [], [], [], []
    for L in (4.0, 8.0, 16.0, 32.0):
        n_along = int(round(L / DY))
        B = circuit_case(L, n_along, Rc=0.0, label=f"bridge L={L:g} m")
        if B is None:
            continue
        mx, rms = deviation(B, femm_welded)
        f = sheet_fraction(L, n_along, 3)
        print(f"    L={L:5.1f} m  f={f:.3f}  ->  max {mx:6.2f} %, RMS {rms:6.2f} %")
        lengths.append(L)
        curves.append(B)
        devs.append((mx, rms))
        fracs.append(f)
    print("  This sweep is EXPECTED to stall, not converge: at fixed dy,"
          "\n  f -> (nx+1)/[(nx+1)+W/dy] independently of L. See T4.")
    return (np.array(lengths), np.array(curves), np.array(devs),
            np.array(fracs))


def run_T3(femm_welded, L=8.0):
    print(f"\n=== T3: cross-section refinement at L = {L:g} m ===")
    n_along = int(round(L / DY))
    n_acrosses, curves, devs, fracs = [], [], [], []
    for na in (3, 5, 9, 15):
        B = circuit_case(L, n_along, n_across=na, Rc=0.0,
                         label=f"refine n_across={na}")
        if B is None:
            continue
        mx, rms = deviation(B, femm_welded)
        f = sheet_fraction(L, n_along, na)
        print(f"    n_across={na:2d}  f={f:.3f}  (1-f)={100 * (1 - f):5.1f} %"
              f"  ->  max {mx:6.2f} %, RMS {rms:6.2f} %")
        n_acrosses.append(na)
        curves.append(B)
        devs.append((mx, rms))
        fracs.append(f)
    print("  If RMS deviation falls roughly in step with (1-f), the residual gap"
          "\n  is the equivalent-wire resistance split, not the mesh resolution.")

    # Are the two convergence axes separable? Refining the cross-section at a
    # longer shield should give the same picture if they are independent.
    B = circuit_case(16.0, int(round(16.0 / DY)), n_across=9, Rc=0.0,
                     label="separability check L=16 n_across=9")
    if B is not None:
        mx, rms = deviation(B, femm_welded)
        print(f"    L=16, n_across=9 -> max {mx:6.2f} %, RMS {rms:6.2f} %")

    return (np.array(n_acrosses), np.array(curves), np.array(devs),
            np.array(fracs))


def run_T4(femm_welded):
    """The corrected bridge: long shield AND f driven towards 1."""
    print("\n=== T4: corrected bridge (large L, f -> 1) ===")
    print("     L    dy  n_ac      f     max %    RMS %")
    params, curves, devs, fracs = [], [], [], []
    for L, dy, n_ac in ((32.0, 1.0, 9), (32.0, 1.0, 15),
                        (64.0, 1.0, 9), (64.0, 2.0, 15)):
        n_along = int(round(L / dy))
        B = circuit_case(L, n_along, n_across=n_ac, Rc=0.0,
                         label=f"corrected L={L:g} dy={dy:g} n_across={n_ac}")
        if B is None:
            continue
        mx, rms = deviation(B, femm_welded)
        f = sheet_fraction(L, n_along, n_ac)
        print(f"  {L:5.0f} {dy:5.2f} {n_ac:5d}  {f:6.3f} {mx:8.2f} {rms:8.2f}")
        params.append((L, dy, n_ac))
        curves.append(B)
        devs.append((mx, rms))
        fracs.append(f)
    return (np.array(params), np.array(curves), np.array(devs),
            np.array(fracs))


def main():
    analytic, femm_b0, t0_dev = run_T0()
    if t0_dev >= 0.5:
        print("\nT0 failed: the FEMM source model is wrong. Stopping -- nothing"
              "\ndownstream would mean anything.")
        return 1

    t1 = run_T1()
    lens, bridge, bridge_dev, bridge_frac = run_T2(t1["femm_welded"])
    nacs, refine, refine_dev, refine_frac = run_T3(t1["femm_welded"])
    t4_par, t4_curves, t4_dev, t4_frac = run_T4(t1["femm_welded"])

    # Pooled across T2/T3/T4: does deviation collapse onto (1 - f)?
    #
    # The L = 4 m case is excluded from the pooled statistic. It is the genuinely
    # finite 4 m x 4 m shield, where the error is dominated by end effects rather
    # than by f -- the one case the relation is not claimed to describe. Leaving
    # it in would let it dominate an origin-forced fit (slope 0.99 with it, 0.64
    # without). It is still plotted, marked as the outlier it is.
    all_f = np.concatenate([bridge_frac, refine_frac, t4_frac])
    all_rms = np.concatenate([bridge_dev[:, 1], refine_dev[:, 1], t4_dev[:, 1]])
    long_enough = np.concatenate([lens >= 8.0,
                                  np.ones(len(refine_frac), bool),
                                  np.ones(len(t4_frac), bool)])
    ratio = all_rms[long_enough] / (100.0 * (1.0 - all_f[long_enough]))
    print(f"\nRMS deviation / (1-f), long-shield cases only:  mean {ratio.mean():.2f}, "
          f"spread {ratio.min():.2f}-{ratio.max():.2f}  ({len(ratio)} cases)")
    excluded = (~long_enough).sum()
    if excluded:
        print(f"  ({excluded} short-shield case excluded: end-effect dominated)")
    best = int(np.argmin(all_rms))
    print(f"best agreement: f={all_f[best]:.3f} -> RMS {all_rms[best]:.2f} %")

    store = {
        "x": X, "analytic": analytic, "femm_b0": femm_b0,
        "femm_welded": t1["femm_welded"], "femm_isolated": t1["femm_isolated"],
        "bridge_lengths": lens, "bridge_curves": bridge, "bridge_dev": bridge_dev,
        "bridge_fraction": bridge_frac,
        "refine_n_across": nacs, "refine_curves": refine, "refine_dev": refine_dev,
        "refine_fraction": refine_frac,
        "t4_params": t4_par, "t4_curves": t4_curves, "t4_dev": t4_dev,
        "t4_fraction": t4_frac,
    }
    for k in ("circuit_rc0", "circuit_rc100"):
        if t1.get(k) is not None:
            store[k] = t1[k]
    np.savez(OUT, **store)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
