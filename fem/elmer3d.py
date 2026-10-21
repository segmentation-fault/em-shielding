"""Independent 3-D FEM reference (Elmer) for the FINITE 4 m x 4 m shield.

The 3-D analogue of ``fem/femm2d.py`` and the second, independent axis of
validation. ``femm2d.py`` checks an *infinitely long* shield in 2-D; here the
shield is genuinely finite (4 m x 4 m), the circuit method's distinguishing case
and the one ``fem/tune.py`` flags as having no independent reference.

Formulation -- imposed analytic source
--------------------------------------
The circuit method's source is three *infinite line currents* with a known
closed-form vector potential. Rather than model finite cables (which cannot
carry an exactly-prescribed balanced 3-phase current in a plain magnetodynamics
solve -- terminal- and body-current drives either scramble the phases or are
gauged away), this model imposes that analytic potential directly:

* The air box is placed so the buried cables lie *below* its bottom face -- the
  interior is therefore source-free, and the infinite-line potential
  A_y(x,z) = -(mu0/2pi) sum_k I_k ln r_k is a harmonic field there.
* A_y is imposed on the outer boundary as the tangential A (a MATC expression
  for the y-component; the x,z tangential parts are zero). In a source-free
  region this Dirichlet data reproduces the exact source field inside, so with
  no plate the model returns the analytic infinite-line field to rounding -- the
  J0 gate is then exact, not merely close.
* The 3 mm plate is a **thin-sheet boundary condition** on a zero-thickness
  4 x 4 surface at z = 0 (skin depth >> 3 mm, so the current is uniform through
  the thickness); it perturbs the interior field, and |B| along the profile is
  the shielded result. This is the same thin-conductive-sheet shielding model as
  the ``bib/`` papers.

This is the most faithful possible comparison to the circuit method: identical
infinite-line source, no finite-cable approximation.

STATUS (2026-08-15): the pipeline, mesh, thin-sheet model and MUMPS solve all
work and are validated (Elmer's own mgdyn_harmonic_wire passes to its reference
norm with this build). The **source imposition is not yet correct**: the no-plate
J0 field comes out a roughly uniform ~24x too small, which points to the
edge-DOF Dirichlet ``A {e} 2 = <analytic A_y>`` not imposing the intended
tangential potential at the expected scale (an Elmer edge-element BC-semantics
issue), rather than a mesh or geometry error. Earlier source models were
eliminated for definite reasons: a body current density is projected to zero by
the solver's source gauge; terminal/potential drives on solid conductors scramble
the three phases' *relative* phase even when solved exactly (measured directly).
The analytic-boundary route was then debugged and abandoned: imposing the source
A_y as a spatially-varying tangential-A Dirichlet condition does not work in
Elmer's edge elements -- the component form ``A re {e} 2 = <MATC>`` is applied
but yields a wrong, x-tilted field, and the vector form ``A re {e} = <3-vector
MATC>`` yields identically zero. Elmer is not evaluating the varying tangential
A per edge. The full Circuits stranded-coil machinery was then built and run (three phased
stranded coils, DirectionSolver/RotM/WPotential chain, per-coil circuit
definitions -- see the scratchpad prototypes coil3_geom.py / coil3_run.py). It
now drives the correct balanced current: the key fix was naming every circuit's
source generically "S" (resolved per coil body), not "S1/S2/S3", which had left
the circuit RHS at zero. With that the J0 source field rose from ~5 % to ~50-60 %
of analytic and took roughly the right shape. The sole remaining blocker is
convergence: the circuit coupling requires the iterative "Circuit" preconditioner
(a MUMPS direct solve *decouples* the circuit -> zero current), and that solve
*diverges* on this air-dominated H(curl) system (A-field norm blows up to ~1e10,
field energy ~1e15, residual climbing after ~3000 iters), so the sampled field is
a fragment of a diverging solution (~2x low, non-monotonic). The
clean unlock is to rebuild this Elmer with **Hypre** and use its AMS
(auxiliary-space Maxwell) preconditioner -- built for exactly this curl-curl
operator -- inside the circuit-coupled solve. Everything upstream of the source
-- mesh, thin-sheet plate, MUMPS solve for the plain field problem, sampling, WSL
pipeline -- is ready and validated.

Solver
------
Harmonic A-V (Whitney edge elements). A robust direct solve needs MUMPS, absent
from the prebuilt Elmer *Windows* binaries; set ELMER_WSL_DISTRO to run a
MUMPS-enabled Elmer built under WSL (see fem/elmer3d.py::_wsl_config and the
project README). The build is validated against Elmer's own mgdyn_harmonic_wire
reference norm.

Coordinate frame matches ``shielding.geometry``: x transverse, y along the
cables, z up, ground at z = 0.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np

MU0 = 4e-7 * np.pi


def find_elmer():
    """Locate ElmerSolver and ElmerGrid for a native (non-WSL) run."""
    candidates = []
    home = os.environ.get("ELMER_HOME")
    if home:
        candidates.append(os.path.join(home, "bin"))
    candidates += [
        r"C:\Users\Admin\ElmerFEM\bin",
        r"C:\Program Files\Elmer 9.0-Release\bin",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Elmer", "bin"),
    ]

    def _which(name):
        p = shutil.which(name)
        if p:
            return p
        for d in candidates:
            for ext in (".exe", ""):
                cand = os.path.join(d, name + ext)
                if os.path.isfile(cand):
                    return cand
        return None

    solver, grid = _which("ElmerSolver"), _which("ElmerGrid")
    if not solver or not grid:
        raise FileNotFoundError(
            "ElmerSolver/ElmerGrid not found. Either set ELMER_WSL_DISTRO to run "
            "a MUMPS-enabled Elmer under WSL, or install a native Elmer and put "
            "its bin/ on PATH / set ELMER_HOME.")
    return solver, grid


def _wsl_config():
    """Return (distro, wsl_elmer_home) if a WSL Elmer is configured, else None.

    Set ELMER_WSL_DISTRO (e.g. "Ubuntu-24.04") to run ElmerGrid/ElmerSolver from
    a Linux Elmer through wsl.exe -- the robust route on Windows, since the
    prebuilt Windows binaries lack MUMPS. ELMER_WSL_HOME is the install prefix
    inside WSL (default "$HOME/elmer").
    """
    distro = os.environ.get("ELMER_WSL_DISTRO")
    if not distro:
        return None
    return distro, os.environ.get("ELMER_WSL_HOME", "$HOME/elmer")


def _to_wsl_path(winpath):
    """C:\\a\\b -> /mnt/c/a/b so WSL can reach a Windows working directory."""
    p = os.path.abspath(winpath)
    return f"/mnt/{p[0].lower()}{p[2:].replace(os.sep, '/')}"


def _wsl_run(distro, home, workdir, shell_cmd):
    """Run one Elmer command inside WSL, cwd at the (translated) workdir.

    Binaries are called by absolute path (shell_cmd already includes {home}/bin);
    PATH is deliberately left untouched -- the Windows PATH WSL inherits contains
    parenthesised entries that break re-export under bash.
    """
    inner = (f"export ELMER_HOME={home}; "
             f"export LD_LIBRARY_PATH={home}/lib/elmersolver:$LD_LIBRARY_PATH; "
             f'cd "{_to_wsl_path(workdir)}" && {shell_cmd}')
    return subprocess.run(["wsl.exe", "-d", distro, "--", "bash", "-lc", inner],
                          capture_output=True, text=True)


class Cables3D:
    """Three infinitely long phase conductors parallel to y, balanced 3-phase.

    Mirrors ``shielding.geometry.Cables`` so the same source drives both the
    circuit method and this model. ``xc``/``zc`` in metres, ``I`` complex A.
    """

    def __init__(self, spacing=0.25, depth=1.5, current=900.0, seq=+1):
        self.xc = np.array([-spacing, 0.0, spacing])
        self.zc = np.array([-depth, -depth, -depth])
        ph = np.array([2 * np.pi / 3, 0.0, -2 * np.pi / 3]) * seq
        self.I = current * np.exp(1j * ph)


class Elmer3D:
    """Build, solve and sample one Elmer harmonic 3-D model of the finite shield.

    with_plate=False drops the plate -- the J0 source gate (must return the
    analytic infinite-line field to rounding).
    """

    def __init__(self, cables, with_plate=True, W=4.0, L=4.0, thickness=3e-3,
                 sigma=5.8e7, freq=50.0, box_x=15.0, box_y=15.0, box_ztop=15.0,
                 box_zbot=None, lc_plate=0.12, lc_far=1.5, lc_sample=0.12,
                 sample_x=4.5, direct=True, workdir=None):
        self.cables = cables
        self.with_plate = with_plate
        self.W, self.L, self.thickness, self.sigma = W, L, thickness, sigma
        self.freq = freq
        self.omega = 2 * np.pi * freq
        self.box_x, self.box_y, self.box_ztop = box_x, box_y, box_ztop
        depth = float(-cables.zc[0])
        # Bottom face just above the buried cables so the interior is source-free
        # (the analytic-potential BC is only valid where there is no source).
        self.box_zbot = box_zbot if box_zbot is not None else -depth + 0.05
        self.lc_plate, self.lc_far, self.lc_sample = lc_plate, lc_far, lc_sample
        self.sample_x = sample_x
        self.direct = direct
        self._workdir = workdir or tempfile.mkdtemp(prefix="elmer3d_")
        os.makedirs(self._workdir, exist_ok=True)
        self.mesh_dir = os.path.join(self._workdir, "mesh")
        self.msh_file = os.path.join(self._workdir, "model.msh")
        self.sif_file = os.path.join(self._workdir, "case.sif")
        self.line_file = os.path.join(self._workdir, "bline.dat")
        self._tags = None
        self._solved = False

    # ------------------------------------------------------------- geometry

    def build_mesh(self):
        """Generate the gmsh mesh (box + embedded plate surface) and write .msh."""
        import gmsh

        W, L = self.W, self.L
        Rx, Ry = self.box_x, self.box_y
        zb, zt = self.box_zbot, self.box_ztop
        eps = 1e-3

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("shield3d")
        occ = gmsh.model.occ

        box = occ.addBox(-Rx, -Ry, zb, 2 * Rx, 2 * Ry, zt - zb)
        if self.with_plate:
            plate = occ.addRectangle(-W / 2, -L / 2, 0.0, W, L)
            occ.fragment([(3, box)], [(2, plate)])
        occ.synchronize()

        air_vol = gmsh.model.getEntities(3)[0][1]
        plate_faces, outer = [], []
        for (_, tg) in gmsh.model.getEntities(2):
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, tg)
            cz = (z0 + z1) / 2
            dx, dz = x1 - x0, z1 - z0
            on = (abs(x0 + Rx) < eps or abs(x1 - Rx) < eps
                  or abs(y0 + Ry) < eps or abs(y1 - Ry) < eps
                  or abs(z0 - zb) < eps or abs(z1 - zt) < eps)
            if self.with_plate and abs(cz) < eps and dx > W - 0.5 and dz < eps:
                plate_faces.append(tg)
            elif on:
                outer.append(tg)
        if self.with_plate and not plate_faces:
            gmsh.finalize()
            raise RuntimeError("plate surface not found after fragment")

        # ---- mesh-size fields ----
        fields = []

        def box_field(vin, xmn, xmx, ymn, ymx, zmn, zmx, thick=0.3):
            f = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(f, "VIn", vin)
            gmsh.model.mesh.field.setNumber(f, "VOut", self.lc_far)
            for key, val in (("XMin", xmn), ("XMax", xmx), ("YMin", ymn),
                             ("YMax", ymx), ("ZMin", zmn), ("ZMax", zmx),
                             ("Thickness", thick)):
                gmsh.model.mesh.field.setNumber(f, key, val)
            fields.append(f)

        if self.with_plate:
            box_field(self.lc_plate, -W / 2 - 0.3, W / 2 + 0.3,
                      -L / 2 - 0.3, L / 2 + 0.3, self.box_zbot, 1.3)
        # Resolve the whole active region between the (close, strong-field)
        # source below box_zbot and the sampling line at z ~ 1: a coarse bulk
        # mesh here leaves the source field unresolved and corrupts the interior
        # solve. Refine a slab from the bottom boundary up past the sample line.
        box_field(self.lc_sample, -1.5, self.sample_x + 0.5, -1.5, 1.5,
                  self.box_zbot, 1.5, thick=0.6)
        mn = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(mn, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(mn)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.lc_far)

        # ---- physical groups ----  air=1; outer=1, [plate=2]
        t = {}
        gmsh.model.addPhysicalGroup(3, [air_vol], 1)
        gmsh.model.setPhysicalName(3, 1, "air")
        t["air"] = 1
        gmsh.model.addPhysicalGroup(2, outer, 1)
        gmsh.model.setPhysicalName(2, 1, "outer")
        t["outer"] = 1
        if self.with_plate:
            gmsh.model.addPhysicalGroup(2, plate_faces, 2)
            gmsh.model.setPhysicalName(2, 2, "plate")
            t["plate"] = 2

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.generate(3)
        gmsh.write(self.msh_file)
        ncoord = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)
        _, etags, _ = gmsh.model.mesh.getElements(3)
        self.mesh_stats = {"nodes": len(ncoord),
                           "elements": int(sum(len(e) for e in etags))}
        self._tags = t
        gmsh.finalize()
        return self

    # ------------------------------------------------------------ sif + run

    def convert_mesh(self):
        """ElmerGrid: gmsh .msh (format 14) -> ElmerSolver mesh (format 2)."""
        wsl = _wsl_config()
        if wsl:
            cmd = (f"{wsl[1]}/bin/ElmerGrid 14 2 "
                   f"{os.path.basename(self.msh_file)} -out mesh -autoclean")
            out = _wsl_run(wsl[0], wsl[1], self._workdir, cmd)
        else:
            _, grid = find_elmer()
            out = subprocess.run(
                [grid, "14", "2", os.path.basename(self.msh_file),
                 "-out", "mesh", "-autoclean"],
                cwd=self._workdir, capture_output=True, text=True)
        if not os.path.isfile(os.path.join(self.mesh_dir, "mesh.header")):
            raise RuntimeError("ElmerGrid did not produce a mesh:\n"
                               + out.stdout[-2000:] + out.stderr[-2000:])
        return self

    def _matc_Ay(self, part):
        """MATC expression for the source A_y (part='re'|'im') vs (tx0=x,tx2=z)."""
        comp = self.cables.I.real if part == "re" else self.cables.I.imag
        terms = []
        for xc, zc, ic in zip(self.cables.xc, self.cables.zc, comp):
            terms.append(f"({ic:.10e})*log(sqrt((tx(0)-({xc:.6f}))^2"
                         f"+(tx(2)-({zc:.6f}))^2))")
        return f"{-MU0 / (2 * np.pi):.10e}*(" + "+".join(terms) + ")"

    def write_sif(self, points):
        """Emit the Elmer harmonic-magnetodynamics deck for a sampling line."""
        pts = np.asarray(points, float)
        x0, x1 = pts[0, 0], pts[-1, 0]
        ys, zs = pts[0, 1], pts[0, 2]
        ndiv = len(pts) - 1
        w = self.omega
        t = self._tags

        if self.direct:
            lin = ('  Linear System Solver = Direct\n'
                   '  Linear System Direct Method = MUMPS\n')
        else:
            lin = ('  Linear System Solver = Iterative\n'
                   '  Linear System Iterative Method = BiCGStabl\n'
                   '  Linear System Max Iterations = 5000\n'
                   '  Linear System Convergence Tolerance = 1.0e-7\n'
                   '  Linear System Abort Not Converged = False\n'
                   '  Linear System Residual Output = 50\n')

        S, A = [], None
        S.append('Check Keywords "Warn"\n')
        S.append('Header\n  Mesh DB "." "mesh"\nEnd\n')
        S.append('Simulation\n  Max Output Level = 4\n'
                 '  Coordinate System = Cartesian\n'
                 '  Simulation Type = Steady State\n'
                 '  Steady State Max Iterations = 1\n  Output Intervals(1) = 0\nEnd\n')
        S.append('Initial Condition 1\n  A re = Real 0\n  A im = Real 0\n'
                 '  A re {e} = Real 0\n  A im {e} = Real 0\nEnd\n')
        S.append('Body 1\n  Name = "air"\n  Equation = 1\n  Material = 1\n'
                 '  Initial Condition = 1\nEnd\n')
        S.append('Material 1\n  Name = "air"\n  Relative Permeability = 1\n'
                 '  Electric Conductivity = 0\nEnd\n')

        S.append('Solver 1\n  Equation = "MGDynamics"\n'
                 '  Variable = A[A re:1 A im:1]\n'
                 '  Procedure = "MagnetoDynamics" "WhitneyAVHarmonicSolver"\n'
                 f'  Angular Frequency = Real {w:.10e}\n'
                 '  Linear System Symmetric = True\n'
                 '  Linear System Complex = Logical True\n'
                 + lin +
                 '  Steady State Convergence Tolerance = 1e-6\nEnd\n')
        S.append('Solver 2\n  Equation = "MGDynamicsCalc"\n'
                 '  Procedure = "MagnetoDynamics" "MagnetoDynamicsCalcFields"\n'
                 '  Potential Variable = String "A"\n'
                 f'  Angular Frequency = Real {w:.10e}\n'
                 '  Calculate Magnetic Flux Density = Logical True\n'
                 '  Linear System Symmetric = True\n'
                 '  Linear System Solver = Iterative\n'
                 '  Linear System Iterative Method = CG\n'
                 '  Linear System Preconditioning = None\n'
                 '  Linear System Max Iterations = 5000\n'
                 '  Linear System Convergence Tolerance = 1.0e-8\n'
                 '  Linear System Residual Output = 0\nEnd\n')
        S.append('Solver 3\n  Exec Solver = After All\n  Equation = "SaveLine"\n'
                 '  Procedure = "SaveData" "SaveLine"\n'
                 f'  Filename = "{os.path.basename(self.line_file)}"\n'
                 '  File Append = False\n'
                 f'  Polyline Coordinates(2,3) = Real {x0:.6f} {ys:.6f} {zs:.6f}'
                 f' {x1:.6f} {ys:.6f} {zs:.6f}\n'
                 f'  Polyline Divisions(1) = Integer {ndiv}\nEnd\n')
        S.append('Equation 1\n  Active Solvers(3) = 1 2 3\nEnd\n')

        # BC 1: outer boundary carries the analytic infinite-line potential
        # (tangential A = A_y y-hat). The x,z tangential parts are zero.
        S.append('Boundary Condition 1\n  Target Boundaries(1) = 1\n'
                 '  A re {e} 1 = Real 0\n  A re {e} 3 = Real 0\n'
                 '  A im {e} 1 = Real 0\n  A im {e} 3 = Real 0\n'
                 '  A re {e} 2 = Variable Coordinate\n'
                 f'    Real MATC "{self._matc_Ay("re")}"\n'
                 '  A im {e} 2 = Variable Coordinate\n'
                 f'    Real MATC "{self._matc_Ay("im")}"\nEnd\n')
        if self.with_plate:
            S.append(f'Boundary Condition 2\n  Target Boundaries(1) = {t["plate"]}\n'
                     f'  Thin Sheet Thickness = Real {self.thickness:.6g}\n'
                     '  Thin Sheet Relative Permeability = Real 1.0\n'
                     f'  Thin Sheet Electric Conductivity = Real {self.sigma:.6g}\nEnd\n')

        with open(self.sif_file, "w") as fh:
            fh.write("\n".join(S))
        return self

    def solve(self, nprocs=1):
        with open(os.path.join(self._workdir, "ELMERSOLVER_STARTINFO"), "w") as fh:
            fh.write(os.path.basename(self.sif_file) + "\n")
        wsl = _wsl_config()
        if wsl:
            home = wsl[1]
            cmd = (f"{home}/bin/ElmerSolver" if nprocs == 1
                   else f"mpirun -np {nprocs} {home}/bin/ElmerSolver_mpi")
            out = _wsl_run(wsl[0], home, self._workdir, cmd)
        else:
            solver, _ = find_elmer()
            env = dict(os.environ)
            home = os.path.dirname(os.path.dirname(solver))
            env.setdefault("ELMER_HOME", home)
            env["PATH"] = os.path.join(home, "bin") + os.pathsep + env.get("PATH", "")
            out = subprocess.run([solver], cwd=self._workdir,
                                 capture_output=True, text=True, env=env)
        self._solver_log = out.stdout + out.stderr
        if not os.path.isfile(self.line_file):
            raise RuntimeError("ElmerSolver produced no line output:\n"
                               + self._solver_log[-3000:])
        self._solved = True
        return self

    def sample_B(self, points):
        """Complex phasor B (N,3, Tesla) at the profile points from SaveLine."""
        if not self._solved:
            raise RuntimeError("call build_mesh/convert_mesh/write_sif/solve first")
        return _read_saveline(self.line_file, np.asarray(points, float))


def _read_saveline(line_file, points):
    """Parse an Elmer SaveLine .dat (+ .dat.names) into complex B at `points`."""
    names_file = line_file + ".names"
    col = {}
    if os.path.isfile(names_file):
        with open(names_file) as fh:
            for ln in fh:
                if ":" not in ln:
                    continue
                idx, _, name = ln.partition(":")
                idx = idx.strip()
                if idx.isdigit():
                    col[name.strip().lower()] = int(idx) - 1
    data = np.loadtxt(line_file)
    if data.ndim == 1:
        data = data[None, :]

    def find(*keys):
        for k in keys:
            if k in col:
                return col[k]
        for name, i in col.items():
            if all(term in name for term in keys[0].split()):
                return i
        raise KeyError(f"SaveLine column not found: {keys}; have {list(col)}")

    cx = find("coordinate 1")
    B = np.zeros((data.shape[0], 3), dtype=complex)
    B[:, 0] = data[:, find("magnetic flux density re 1", "magnetic flux density 1")] \
        + 1j * data[:, find("magnetic flux density im 1")]
    B[:, 1] = data[:, find("magnetic flux density re 2", "magnetic flux density 2")] \
        + 1j * data[:, find("magnetic flux density im 2")]
    B[:, 2] = data[:, find("magnetic flux density re 3", "magnetic flux density 3")] \
        + 1j * data[:, find("magnetic flux density im 3")]

    xs = data[:, cx]
    order = np.argsort(xs)
    xs, B = xs[order], B[order]
    xq = points[:, 0]
    if len(xs) == len(xq) and np.allclose(xs, np.sort(xq), atol=1e-6):
        return B
    out = np.zeros((len(xq), 3), dtype=complex)
    for c in range(3):
        out[:, c] = (np.interp(xq, xs, B[:, c].real)
                     + 1j * np.interp(xq, xs, B[:, c].imag))
    return out


def elmer_norm(cables, points, with_plate=True, **kw):
    """|B| in microtesla at `points` for one Elmer case; returns (norm, model)."""
    from shielding.field import field_norm
    m = Elmer3D(cables, with_plate=with_plate, **kw)
    m.build_mesh().convert_mesh().write_sif(points).solve()
    return field_norm(m.sample_B(points)) * 1e6, m


def analytic_line_B(cables, points):
    """Complex phasor field of the infinite-line cables (Tesla), (N,3).

    Duplicates ``shielding.field.cable_field`` so ``fem/compare3d.py`` gates the
    3-D source against the same reference the 2-D T0 gate uses.
    """
    pts = np.asarray(points, float)
    B = np.zeros((pts.shape[0], 3), dtype=complex)
    for xc, zc, I in zip(cables.xc, cables.zc, cables.I):
        dx = pts[:, 0] - xc
        dz = pts[:, 2] - zc
        r2 = dx * dx + dz * dz
        pref = MU0 / (2 * np.pi) * I / r2
        B[:, 0] += pref * dz
        B[:, 2] += -pref * dx
    return B
