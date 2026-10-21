# 3-D Elmer circuits stranded-coil source — work in progress

These are the working prototypes for the Elmer 3-D validation's source model,
kept out of the main `fem/elmer3d.py` because the coupled solve does not yet
converge (see below). They require the WSL MUMPS/Hypre Elmer build and run
through `fem.elmer3d._wsl_run`.

- `coil_geom.py` / `coil_run.py` — a single straight stranded coil (diagnostic).
- `coil3_geom.py` / `coil3_run.py` — the three phased stranded coils, the actual
  balanced 3-phase source. `coil3_geom` tags the 6 faces of each cable
  (gamma/alpha/beta ends) for the DirectionSolver/WPotential chain; `coil3_run`
  builds the circuit definitions, the full solver chain (DirectionSolver ×2,
  RotMSolver, WPotentialSolver, CircuitsAndDynamicsHarmonic, WhitneyAVHarmonic,
  CalcFields, CircuitsOutput, SaveLine) and drives each coil at its phase.

## State

- The coils **carry the correct balanced current** once every circuit's source is
  named generically `"S"` (resolved per coil body) rather than `S1/S2/S3` — the
  numbered form left the circuit RHS at zero. With `"S"`, the no-plate J0 source
  field reaches ~50–60 % of the analytic infinite-line field with roughly the
  right shape.
- **Blocker:** the circuit coupling requires the iterative `Circuit`
  preconditioner (a MUMPS *direct* solve decouples the circuit → zero current),
  and that iterative `WhitneyAVHarmonicSolver` solve **diverges** on this large,
  mostly-insulating air domain — the A-field norm blows up (~1e10, field energy
  ~1e15) and the residual climbs after ~3000 iterations. The sampled field is a
  fragment of that diverging solution, hence ~2× low and non-monotonic.

## The unlock

A robust solver for the circuit-coupled H(curl) system. Two routes:

1. **Hypre AMS.** Rebuild the WSL Elmer with Hypre (`apt install libhypre-dev`,
   reconfigure `-DWITH_Hypre=TRUE`, rebuild) and use its AMS (auxiliary-space
   Maxwell) preconditioner — designed for exactly this curl-curl operator —
   inside the circuit-coupled solve. Needs confirming AMS composes with the
   `Circuit` preconditioner.
2. **Monolithic direct circuit solve.** Get MUMPS to factor the *augmented*
   (field + circuit) matrix in one shot instead of decoupling it. This would
   reuse the already-built MUMPS and avoid a rebuild, but needs the right Elmer
   keywords to assemble the circuit rows into the direct system.

Everything upstream — gmsh meshing, the thin-sheet plate model, the plain-field
MUMPS solve (validated against Elmer's own `mgdyn_harmonic_wire` reference norm),
sampling, and the Windows→WSL pipeline in `fem/elmer3d.py` — is ready.
