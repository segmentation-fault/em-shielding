# Fast (non-FEM) magnetic shielding of buried HV cables

A compact Python implementation of the **wire-filament circuit method** for
computing the magnetic shielding effectiveness of conductive plates placed over
buried three-phase HV cables — a fast, mesh-free alternative to FEM.

The method is the one shared by three papers:

- Clairmont & Lordan, *IEEE T-PWRD* 14(4), 1999 — the foundational technique and
  the closed-form self/mutual inductance formulas.
- Sergeant, Dupré & Melkebeek, *COMPEL* 27(1), 2008 — the buried-cable
  application (fully specified geometry, richest set of plots).
- De Wulf et al., *JMMM* 316, 2007 — the same "multiconductor" method.

Two further papers place this work in context rather than being
reproduced:

- Xia & Du, *IEEE T-EMC* 56(3), 2014 — the ferromagnetic PEEC extension of the
  same circuit family (carries magnetization **M** with a double-exponential
  through-thickness profile). Its Case B (1 m × 4 m plate, 40 × 160 cells
  compared against a 2-D FEM reference, ~1.5% agreement) is an independent
  external check on the L ≫ W regime `fem/tune.py` characterizes — see that
  section.
- Etse et al., *IEEE T-EMC* 65(5), 2023 — thin-layer FEM discretization (AMSL /
  INBC homogenization of multilayer composites). Relevant to the FEM *reference*
  side, not to the circuit method; its "refine until ΔSE < 10⁻³ dB" convergence
  discipline is the same logic the FEMM mesh-convergence work here applies.

## Why this method

Each thin conductive plate is "re-melted" into a rectangular grid of straight
wire filaments. Every filament gets a resistance, a self-inductance, and a mutual
inductance to every other filament and to the source cables. Kirchhoff's laws
give a small dense complex linear system for the induced (eddy) currents;
Biot-Savart then gives **B** anywhere as the superposition of source-cable and
shield-current fields. No volume mesh, no open-boundary truncation, and — unlike
2-D FEM — it captures the finite plate length that dominates this geometry.
Solve time is well under a second for the standard 8-plate shield.

Valid for non-ferromagnetic plates (Al, Cu) at power frequency with thickness
much smaller than the skin depth. Ferromagnetic (flux-shunting) shields are out
of scope; they can only be approximated via an equivalent conductivity. Xia & Du
(*T-EMC* 56(3), 2014) show how the same circuit family is extended to
that case, by carrying a magnetization unknown alongside the induced current.

## Layout

```
shielding/
  inductance.py   self-inductance (Eq.3) and parallel-filament mutual (Neumann)
  geometry.py     cables + multi-plate filament mesh (welded or contact-resistance)
  circuit.py      assemble R + jwL, incidence, source EMF; solve for currents
  field.py        infinite-line cable field, finite-segment Biot-Savart, B0
  analysis.py     high-level helpers (shielded/unshielded B profiles)
fem/
  femm2d.py       independent FEMM 4.2 harmonic 2-D model (geometry only, no physics)
  compare.py      T0-T4 cross-validation harness -> fem/results.npz
  tune.py         parameter study + reference noise floor -> fem/tune.npz
figures/          one driver per reproduced figure -> PNG next to the script
tests/            inductance unit tests, sanity + claim verification
```

## Reproduced figures

| Script | Reproduces |
|---|---|
| `figures/fig_sergeant_05.py` | Sergeant Fig. 5 — B(x) at z=1 m, parallel/transversal x Rc |
| `figures/fig_sergeant_03.py` | Sergeant Fig. 3 — grid-refinement convergence at P2(0,0,1) |
| `figures/fig_sergeant_06_07.py` | Sergeant Fig. 6 & 7 — shield size, width, depth, spacing |
| `figures/fig_dewulf_03.py` | De Wulf Fig. 3 — six-plate open shield vs Rcont |
| `figures/fig_fem_compare.py` | FEM cross-check — finite shield vs 2-D FEM |
| `figures/fig_fem_bridge.py` | FEM cross-check — where the residual gap comes from |
| `figures/fig_fem_tune.py` | parameter tuning — the reference's noise floor, and the cross-section optimum |
| `figures/fig_tuned_claims.py` | the reproduced claims across the mesh family — which survive |
| `figures/fig_tuned_profiles.py` | Sergeant Fig. 5 and De Wulf Fig. 3 with their mesh-uncertainty bands |

## Run

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
.venv/Scripts/python tests/test_inductance.py             # unit tests
.venv/Scripts/python tests/verify_claims.py               # headline claims
.venv/Scripts/python figures/fig_sergeant_05.py           # -> figures/*.png
```

The FEM cross-validation additionally needs FEMM 4.2 installed
(https://www.femm.info/wiki/Download); `pyfemm` is only a COM wrapper.

```bash
.venv/Scripts/python tests/verify_fem_b0.py    # gate: FEM vs analytic source
.venv/Scripts/python fem/compare.py            # T0-T4 -> fem/results.npz
.venv/Scripts/python figures/fig_fem_compare.py
.venv/Scripts/python figures/fig_fem_bridge.py

.venv/Scripts/python fem/tune.py --quick       # parameter study, smoke subset
.venv/Scripts/python fem/tune.py               # full study -> fem/tune.npz (~30 min)
.venv/Scripts/python figures/fig_fem_tune.py

.venv/Scripts/python figures/fig_tuned_claims.py       # ~6 min, solves directly
.venv/Scripts/python figures/fig_tuned_profiles.py     # ~4 min, solves directly
```

## Verified against the papers

- 8x8 welded shield -> field reduced ~6x (Sergeant: "up to six times").
  **Mesh-dependent — see "Which reproduced claims survive the mesh" below.**
- High contact resistance (Rc>=10 mOhm) caps shielding near SF~1.1 (never > 2x).
- Contact resistance monotonically degrades shielding (SF 4.0 -> 1.1).
- Parallel config slightly better than transversal at intermediate Rc.
- n_b=1 meshes converge to a wrong (too-high) value; n_b>=3 converge correctly;
  transversal needs larger n_l — all as in Sergeant Fig. 3.

**Amplitude note — resolved.** The unshielded field here is B0(0,0,1) = 12.37 uT
against the papers' ≈14 uT. That ~12% gap is *not* an error in this code: an
independent FEMM 2-D model of the same source geometry returns 12.34 uT, agreeing
with the analytic infinite-line result to 0.25% RMS across the whole profile
(`tests/verify_fem_b0.py`). Two independent implementations reproduce 12.37 uT, so
≈14 uT is not obtainable from the Table I parameters as stated — most likely a
plotting or digitization difference in the papers.

## FEM cross-validation

Everything above is the circuit method checked against itself and against curves
read off the papers. `fem/` adds an independent check: a FEMM 4.2 harmonic 2-D
model in which we declare only geometry, materials and circuit constraints —
FEMM owns the physics, so it can actually falsify this implementation.

**Scope.** The check is 2-D only. FEMM's planar model makes the shield infinitely
long, so the finite-length 3-D result — the circuit method's distinguishing
feature — is *not* directly validated. What is validated is everything except the
end effects, in the L → ∞ limit.

**3-D axis (Elmer) — work in progress.** `fem/elmer3d.py` + `fem/compare3d.py`
are the intended independent 3-D check of the finite 4 m × 4 m shield (the case
this 2-D model structurally cannot reach). Built and validated so far: a
MUMPS-enabled Elmer compiled from source under WSL (checked against Elmer's own
`mgdyn_harmonic_wire` reference norm), the gmsh → ElmerGrid → ElmerSolver → sample
pipeline driven from Windows through `wsl.exe`, and the plate modelled as a
thin-sheet boundary condition. The open piece is the 3-phase source: the
`fem/circuits_wip/` stranded-coil prototype drives the correct balanced current
but its circuit-coupled H(curl) solve diverges with the available preconditioner;
the fix is Hypre AMS (rebuild) or a monolithic direct circuit solve. See the
`STATUS` block in `fem/elmer3d.py` and `fem/circuits_wip/README.md`. Until that
lands, `compare3d.py` produces no valid number — it is scaffolding only. To run
the WSL Elmer once available: set `ELMER_WSL_DISTRO=Ubuntu-24.04`.

**T0 — source model.** With no shield, FEMM reproduces the analytic infinite-line
field to 0.41% max / 0.25% RMS. This is what resolves the amplitude note above.
(It only does so after refining the air mesh along the sampling path: FEMM uses
first-order triangles, so B = curl A is element-wise constant, and the automatic
mesh gives 18% point-sampling error. See `refine_box` in `fem/femm2d.py`.)

**T1 — finite vs infinite.** At x = 0, z = 1 m, against 12.37 uT unshielded:

| | B (uT) | shielding factor |
|---|---|---|
| FEMM 2-D, welded (infinite shield) | 1.03 | 12.0 |
| circuit method, 4 m x 4 m, Rc = 0 | 3.11 | 4.0 |
| FEMM 2-D, isolated plates | 10.48 | 1.18 |
| circuit method, 4 m x 4 m, Rc = 100 mOhm | 11.36 | 1.09 |

The isolated pair agrees to 8%; the welded pair differs by 200%. That asymmetry
is the clue — see below.

**T2/T3/T4 — the residual gap, and what it is.** Simply lengthening the shield
does *not* converge onto the FEMM curve; it plateaus near 25% RMS. The reason is
`_equiv_wire`: it conserves the plate's metal volume across *both* filament
directions, so for the essentially unidirectional current here the mesh presents
only

    f = (nx+1) L / [ (nx+1) L + (ny+1) W ]

of the true sheet conductance sigma*t — about **0.60** at Sergeant's standard
n_across = 3. Worse, with the along-cable spacing dy held fixed, ny+1 ~ L/dy and
so f → (nx+1)/[(nx+1) + W/dy], which is **independent of L**. No amount of length
can remove it.

Pooled over the 11 long-shield configurations the RMS deviation collapses onto
the shortfall, RMS ≈ 0.64 (1 − f), spread 0.53–1.02. (The 12th case, the
genuinely finite 4 m x 4 m shield, is excluded: its error is dominated by end
effects rather than by f, and it is the one case this relation does not claim to
describe.) Driving f → 1 while keeping the shield long:

| L (m) | dy (m) | n_across | f | RMS dev |
|---|---|---|---|---|
| 8 | 0.25 | 3 | 0.602 | 27.3% |
| 32 | 1.00 | 9 | 0.947 | 3.2% |
| 32 | 1.00 | 15 | 0.967 | 2.6% |
| 64 | 2.00 | 15 | 0.983 | **1.1%** |

**So the circuit method agrees with an independent FEM solver to ~1% in the
regime where both solve the same problem.** The discrepancy at default settings
is not a bug in this code and not a failure of the method — it is a known,
computable property of the Clairmont–Lordan equivalent-wire prescription, which
is faithful here (`_equiv_wire` is unmodified). It matters when the shield is
*resistance*-limited (welded, large eddy currents) and barely matters when it is
*constraint*-limited (isolated plates), which is exactly the T1 asymmetry.

Practical consequence: for a welded shield, Sergeant's n_across = 3 with a fine
along-cable mesh under-predicts shielding substantially. Prefer a cross-section
fine relative to the along-cable spacing, or correct the filament resistance for
the intended current direction.

## Parameter tuning (`fem/tune.py`)

The section above compares at one hand-picked mesh and attributes the whole
residual to the conductance shortfall `(1 - f)`. `fem/tune.py` sweeps the
parameters properly, with error bars. It revises both halves of that.

**The reference has a noise floor, and it is not small.** FEMM meshes with an
unstructured triangulator, so changing any input regenerates the whole mesh and
the samples land in different first-order elements. Re-solving *identical
physics* over seven mesh seeds gives 0.18% RMS scatter between single runs
(worst point 0.79% peak-to-peak). That is the same size as the differences the
T2/T3/T4 tables are built on. Consequences:

- `boundary_radius` "moving" the answer by 0.5% is noise — as physics demands,
  since a balanced three-phase source falls off like 1/r².
- Refining `plate_mesh` buys no accuracy and *costs* reproducibility: reseed
  scatter climbs 0.105% → 0.862% as it goes 1.5 mm → 0.5 mm (sliver elements in
  a 3 mm × 4 m strip). `cable_mesh` is a bit-exact null, as it must be at σ = 0.
- Self-convergence is not accuracy. The cable-outline knob `arc_seg` converges
  monotonically *away* from the truth. With no shield the source is three
  infinite line currents, which `shielding/field.py` gives in closed form, so
  `tune.py` calibrates against that instead: `arc_seg = 5` beats both 10 and 2.

Every FEMM default in this repo is already at or indistinguishable from the
optimum. What was missing was not tuning but an error bar: **δ_FEM = 0.07% RMS**
for a 7-seed ensemble mean, over a **source-model floor of 0.16% RMS**. No
circuit-vs-FEMM claim below that floor means anything.

**The dominant circuit-side axis is the cross-section, not `f`.** `_equiv_wire`
ties the wire radius to the mesh — `r ~ 1/√nx` against a pitch `dx ~ 1/nx` — so
refining the cross-section makes filaments *relatively fatter*, `2r/dx ~ √nx`.
Two opposite-signed errors result, and they cross:

| n_across | 2r/dx | mean signed dev | RMS |
|---|---|---|---|
| 3 | 0.15 | **+7.78%** | 18.69% |
| 9 | 0.26 | +0.23% | 2.68% |
| 21 | 0.40 | −0.53% | **0.63%** |
| 31 | 0.49 | −0.58% | 0.65% |
| 61 | 0.68 | **−1.23%** | 1.96% |

Too few filaments under-shield; overlapping ones over-shield. Refining past
`2r/dx ≈ 0.4` makes agreement *worse* while `1 - f` keeps improving, which no
conductance-only account can produce. At matched `1 - f ≈ 0.007`, changing only
`n_across` moves RMS from 8.10% to 0.65% — so `RMS ≈ 0.64 (1 - f)` above does
not survive: T3 and T4 refine `n_across`, and the gain they credit to `f` is
mostly transverse resolution.

**Tuned settings** (welded, vs the calibrated reference):

| L (m) | dy (m) | n_across | RMS | cost |
|---|---|---|---|---|
| 64 | 2 | 15 | 1.12% | 23 s | 
| 128 | 16 | 21 | **0.63%** | **5 s** |
| 256 | 8 | 21 | 0.57% | 56 s |

The first row is the T4 winner above; half the error for a fifth of the time is
available by moving effort from along-cable refinement to the cross-section.

**What the residual actually is.** With the cross-section at its optimum and
L ≥ 64 m, the deviation is ≈ −0.5% and barely moves while `1 - f` varies 15× and
`W/L` varies 4×; a fitted `c + a(1-f)^α + b(W/L)^β` reduces the residual only
from 0.211 to 0.147 over a plain constant. The isolated-plate regime — which
does not depend on the filament resistance at all — sits at −0.64%, flat in L.
Two physically different regimes showing the same offset is not a property of
the equivalent-wire resistance split; it is a common-mode floor a few times the
reference's own 0.16% source error. The honest statement is that circuit and
FEMM agree to about half a percent, and that this is where the *reference*
stops resolving, not the method.

**An external anchor for this regime.** The finite-geometry agreement here has
no independent reference — the FEMM model is the only cross-check, so the band is
a sensitivity measurement, not an error bar. The closest external anchor is Xia
& Du (2014) Case B: an independent PEEC-vs-2-D-FEM comparison on a 1 m × 4 m
plate (40 × 160 cells) in this same L ≫ W regime, reporting ~1.5% agreement.
That their circuit-vs-FEM residual lands in the same half-to-one-percent band —
with the mesh deliberately far finer along the plate than across it, matching the
transverse-resolution finding above — corroborates the picture from outside this
repo.

**Scope: none of this transfers to the papers' own geometry.** The above holds
in the L ≫ W limit, where `1 - f ≈ 0.002` is negligible and the transverse
resolution dominates. The papers use *square* shields, and there
`f = (nx+1)L / [(nx+1)L + (ny+1)W]` behaves completely differently — for W = L
on a square mesh it is **identically 1/2**, at every refinement level. It is
then not a discretization error that refining removes but a modelling error that
refining preserves, and the T2/T3/T4 account above is right about it. Both
statements are true in their own regime and neither generalises.

## Which reproduced claims survive the mesh (`figures/fig_tuned_claims.py`)

Applying the tuned cross-section to the papers' geometry does not give a better
number; it shows that several of the numbers were never determined. `1 - f` can
only be reduced by making the mesh strongly anisotropic, but a finite shield
carries closed current loops whose return legs live on the along-shield
filaments, so starving `n_along` to buy `f` destroys the loops. On the 4 × 4,
driving `1 - f` from 0.41 to 0.01 takes SF from 4.0 *down* to 1.5.

Sweeping the whole `(n_across, n_along)` family at `P2(0,0,1)`:

| geometry | SF at the published mesh | SF across the family |
|---|---|---|
| 4 m × 4 m welded | 3.97 | 1.7 – 4.9 |
| 8 m × 8 m welded | 6.05 | 3.2 – 13.9 |
| `Rc` ≥ 1 mΩ | 1.09 – 1.34 | ±10% |

So of the claims in the list above:

- "8×8 welded → field reduced ~6×, matching Sergeant's *up to six times*" is
  **not robust**. It holds at `n_across = 3` and nowhere else in the family;
  refining the cross-section at Sergeant's `dy` gives ~14×.
- "welded 4×4 gives a few-fold reduction" is robust as a statement about order
  of magnitude, not as the number 3.97.
- Every contact-resistance claim is robust. That regime is constraint-limited,
  so the resistance model barely enters — the same welded/isolated asymmetry
  `fem/tune.py` measures.

`figures/fig_tuned_profiles.py` redraws Sergeant Fig. 5 and De Wulf Fig. 3 with
the resulting band instead of a single curve: 18–27% wide at low `Rc`, 4% at
high `Rc`. Closing it needs a directional fix to `_equiv_wire`, not tuning.

## Key modeling choices

- Cables are treated as infinitely long lines (as the papers do); their vector
  potential drives the mesh EMFs, and the arbitrary ln reference cancels in KVL.
- Only parallel filaments couple: two perpendicular straight filaments have zero
  mutual inductance, so the coplanar axis-aligned mesh needs only the
  well-behaved parallel-filament formula (validated against Neumann quadrature).
- Contact resistance: Rc=0 -> a single welded conforming mesh; Rc>0 -> plates
  meshed independently and stitched by resistors between coincident edge nodes.
