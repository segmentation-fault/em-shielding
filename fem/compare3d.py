"""J0-J1: cross-validate the filament circuit method against Elmer 3-D.

WORK IN PROGRESS -- this harness is the intended entry point, but it cannot yet
produce a valid number: the Elmer 3-D source model in fem/elmer3d.py is not
finished (see its module docstring "STATUS"). The correct-current source lives in
fem/circuits_wip/ but its coupled solve does not converge. Do not treat J1 output
as a result until the source is fixed. Kept here so the comparison scaffolding is
ready the moment it is.

This is the second, independent axis of validation (see fem/elmer3d.py). The 2-D
FEMM check (fem/compare.py) can only reach the infinitely long shield; here the
shield is the genuinely finite 4 m x 4 m welded plate -- the circuit method's
distinguishing case, and the one fem/tune.py flags as having no independent
reference. Elmer supplies that reference.

    .venv/Scripts/python fem/compare3d.py            # J0 + J1
    .venv/Scripts/python fem/compare3d.py --converge # + box/mesh convergence

Elmer solves are serial and minutes each, so the sweep is deliberately small.

  J0  Source gate, no plate: Elmer vs the analytic infinite-line field. A gate --
      if this fails nothing below means anything (same role as T0).

  J1  The finite 4 m x 4 m welded shield: Elmer 3-D vs the circuit method's own
      `circuit_rc0` curve from fem/results.npz. This is the headline number --
      the first independent check of the finite-geometry shielding factor.

  Jc  (optional) Convergence: refine the truncation box and the sampling-slab
      mesh; |B| at P2 must stop moving. Etse et al.'s "refine until it stops"
      discipline, applied to this reference.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem.elmer3d import Cables3D, Elmer3D, analytic_line_B          # noqa: E402
from fem.compare import deviation                                   # noqa: E402
from shielding.field import field_norm                             # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elmer3d.npz")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.npz")

# Same source + profile as fem/compare.py (Sergeant Table I).
CAB = Cables3D(spacing=0.25, depth=1.5, current=900.0)
X = np.linspace(0.0, 4.0, 21)
PTS = np.column_stack([X, np.zeros_like(X), np.ones_like(X)])

# The one documented working point; the convergence pass varies it. (Keys match
# the current Elmer3D signature; see also ELMER_WSL_DISTRO for the WSL/MUMPS run.)
BASE = dict(box_x=8.0, box_y=8.0, box_ztop=8.0, box_zbot=-1.0,
            lc_far=1.2, lc_plate=0.12, lc_sample=0.13, sample_x=4.5)


def solve(with_plate, **over):
    kw = dict(BASE)
    kw.update(over)
    m = Elmer3D(CAB, with_plate=with_plate, **kw)
    m.build_mesh().convert_mesh().write_sif(PTS).solve()
    B = field_norm(m.sample_B(PTS)) * 1e6
    return B, m.mesh_stats


def run_J0():
    print("\n=== J0: source model, no plate (Elmer vs analytic) ===")
    analytic = field_norm(analytic_line_B(CAB, PTS)) * 1e6
    elmer, stats = solve(with_plate=False)
    mx, rms = deviation(elmer, analytic)
    print(f"  mesh {stats}")
    print(f"  analytic B0(0,0,1) = {analytic[0]:.3f} uT")
    print(f"  Elmer    B0(0,0,1) = {elmer[0]:.3f} uT")
    print(f"  max dev {mx:.2f} %, RMS dev {rms:.2f} %"
          f"   -> {'PASS' if mx < 3.0 else 'FAIL'}")
    return analytic, elmer, mx


def run_J1():
    print("\n=== J1: finite 4 m x 4 m welded shield (Elmer 3-D vs circuit) ===")
    res = np.load(RESULTS)
    circ = res["circuit_rc0"]
    analytic = res["analytic"]
    elmer, stats = solve(with_plate=True)
    mx, rms = deviation(elmer, circ)
    sf_circ = analytic[0] / circ[0]
    sf_elmer = analytic[0] / elmer[0]
    print(f"  mesh {stats}")
    print(f"  circuit  B(P2) = {circ[0]:.3f} uT   SF = {sf_circ:.2f}")
    print(f"  Elmer 3D B(P2) = {elmer[0]:.3f} uT   SF = {sf_elmer:.2f}")
    print(f"  Elmer vs circuit: max {mx:.1f} %, RMS {rms:.1f} %")
    return circ, elmer, (mx, rms), (sf_circ, sf_elmer)


def run_Jc():
    print("\n=== Jc: convergence (box size, sampling-slab mesh) ===")
    rows = []
    base_B, _ = solve(with_plate=False)
    print(f"  base (box=15, lc_sample=0.12): B(P2) = {base_B[0]:.3f} uT")
    for label, over in (("box=20", dict(box_x=20, box_y=20, box_z=20)),
                        ("lc_sample=0.08", dict(lc_sample=0.08))):
        B, _ = solve(with_plate=False, **over)
        d = 100.0 * abs(B[0] - base_B[0]) / base_B[0]
        print(f"  {label:16s}: B(P2) = {B[0]:.3f} uT   d(P2) = {d:.2f} %")
        rows.append((label, B[0], d))
    return base_B, rows


def main():
    converge = "--converge" in sys.argv
    analytic, j0_elmer, j0_mx = run_J0()
    if j0_mx >= 3.0:
        print("\nJ0 failed: the 3-D source model is wrong. Stopping.")
        return 1
    circ, j1_elmer, j1_dev, sfs = run_J1()
    store = dict(x=X, analytic=analytic, j0_elmer=j0_elmer,
                 circuit_rc0=circ, j1_elmer=j1_elmer,
                 j1_dev=np.array(j1_dev), sf=np.array(sfs))
    if converge:
        _, rows = run_Jc()
        store["jc_labels"] = np.array([r[0] for r in rows])
        store["jc_B"] = np.array([r[1] for r in rows])
        store["jc_dev"] = np.array([r[2] for r in rows])
    np.savez(OUT, **store)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
