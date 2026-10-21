"""FEMM 4.2 harmonic 2-D reference model for the buried-cable shielding problem.

This is the *independent* half of the cross-check: here we declare only geometry,
materials and circuit constraints -- FEMM owns the physics. Nothing in this file
reuses the circuit method's formulation, which is the whole point (a solver that
shares my assumptions cannot falsify them).

Coordinate mapping
------------------
The project frame is (x transverse, y along the cables, z up, ground at z = 0).
FEMM is planar with current flowing out of plane, so its plane is our (x, z)
cross-section and its out-of-plane axis is our +y:

    FEMM x  <->  our x        FEMM y  <->  our z        out-of-plane  <->  our y

The model is therefore y-invariant, i.e. the shield is *infinitely long*. That is
exactly the 2-D idealisation Sergeant et al. criticise, and it is why the results
here must not be overlaid naively on the finite-shield curves -- see fem/compare.py.

Contact resistance in 2-D
-------------------------
Under y-invariance all current flows along y, so there is no transverse conduction
path at all and "plates touching or not" is not a physical distinction. What does
distinguish the cases is how many zero-net-current constraints are imposed:

    mode="welded"   -- one circuit spanning all plates: a single dV/dy across the
                       full width, current free to redistribute over it   (Rc = 0)
    mode="isolated" -- one circuit per plate: an independent dV/dy per plate, each
                       constrained to zero net current              (Rc = infinity)

Both use identical geometry and mesh, so the two curves differ only by the
constraint granularity. Intermediate Rc is a 3-D return-path effect and has no
2-D representation; these two curves bracket it.
"""

import os
import tempfile

import numpy as np

try:
    import femm
except ImportError as exc:                                    # pragma: no cover
    raise ImportError(
        "pyfemm is not installed. Run:\n"
        "    .venv/Scripts/python -m pip install pyfemm pywin32"
    ) from exc

AIR = "SoilAir"
CABLE = "CableConductor"
PLATE = "PlateMetal"
NO_CIRCUIT = "<None>"


class Shield:
    """A row of coplanar conductive plates lying in the ground plane.

    width     : total shield width across x (m)
    n_plates  : plates the width is divided into (Sergeant: 8 x 0.5 m)
    thickness : plate thickness (m); the row is centred on z = 0
    sigma     : conductivity (S/m)
    mode      : "welded" (one circuit) or "isolated" (one circuit per plate)
    """

    def __init__(self, width=4.0, n_plates=8, thickness=3e-3,
                 sigma=5.8e7, mode="welded", center_x=0.0):
        if mode not in ("welded", "isolated"):
            raise ValueError(f"unknown shield mode {mode!r}")
        self.width = width
        self.n_plates = n_plates
        self.thickness = thickness
        self.sigma = sigma
        self.mode = mode
        self.center_x = center_x

    @property
    def plate_width(self):
        return self.width / self.n_plates

    def plate_bounds(self, k):
        x0 = self.center_x - self.width / 2.0
        return x0 + k * self.plate_width, x0 + (k + 1) * self.plate_width

    def circuit_names(self):
        if self.mode == "welded":
            return ["shield"]
        return [f"shield{k}" for k in range(self.n_plates)]

    def circuit_of(self, k):
        return "shield" if self.mode == "welded" else f"shield{k}"


class Femm2D:
    """Build, solve and sample one FEMM harmonic model.

    FEMM is a single global out-of-process COM server, so only one Femm2D may be
    live at a time. Use it as a context manager so the instance is always closed.
    """

    def __init__(self, cables, shield=None, freq=50.0, cable_radius=0.02,
                 boundary_radius=25.0, plate_mesh=1.5e-3, cable_mesh=5e-3,
                 air_mesh=0.03, refine_box=(-1.0, 0.2, 5.0, 1.8),
                 precision=1e-8, min_angle=30, arc_seg=5,
                 workdir=None):
        self.cables = cables
        self.shield = shield
        self.freq = freq
        self.cable_radius = cable_radius
        self.boundary_radius = boundary_radius
        self.plate_mesh = plate_mesh
        self.cable_mesh = cable_mesh
        # Solver/discretization controls, exposed so fem/tune.py can measure the
        # reference model's own convergence. The defaults are the values this
        # class used before they were parameters, so every existing result
        # reproduces unchanged.
        self.precision = precision
        self.min_angle = min_angle
        self.arc_seg = arc_seg
        # FEMM uses first-order triangles, so B = curl A is constant within each
        # element. Point samples of B are therefore only as good as the local
        # mesh, and the automatic air mesh is far too coarse along the
        # measurement line. refine_box (xmin, zmin, xmax, zmax) is an air region
        # covering the sampling path, meshed at air_mesh. It is a pure
        # discretization control: A is continuous across its edges, so it adds
        # no physics.
        self.air_mesh = air_mesh
        self.refine_box = refine_box
        self._workdir = workdir or tempfile.mkdtemp(prefix="femm2d_")
        self.femfile = os.path.join(self._workdir, "model.fem")
        self._solved = False

    # ---------------------------------------------------------------- build

    def build(self):
        femm.openfemm(1)                       # 1 = keep the GUI hidden
        femm.newdocument(0)                    # 0 = magnetics
        # freq, units, type, precision, depth, min angle
        femm.mi_probdef(self.freq, "meters", "planar", self.precision, 1.0,
                        self.min_angle)

        self._add_materials()
        self._add_cables()
        if self.shield is not None:
            self._add_shield()
        self._add_refine_box()
        self._add_air_label()

        # Open boundary. Dirichlet (bc=0) is appropriate: the source is balanced
        # three-phase, so the total current is zero and the field falls off like
        # 1/r^2 -- a truncation at 25 m is far outside anything we sample.
        femm.mi_makeABC(7, self.boundary_radius, 0.0, -0.5, 0)
        femm.mi_saveas(self.femfile)
        return self

    def _add_materials(self):
        # name, mu_x, mu_y, H_c, J, Cduct[MS/m], Lam_d, Phi_hmax, lam_fill,
        # LamType, Phi_hx, Phi_hy, NStrands, WireD
        femm.mi_addmaterial(AIR, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
        # Soil is modelled as air (mu_r = 1, sigma = 0), as all three papers do.

        # The cables are prescribed-current sources, so sigma = 0: this suppresses
        # skin/proximity redistribution inside them and makes their external field
        # exactly that of a line current, matching shielding.field.cable_field.
        femm.mi_addmaterial(CABLE, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)

        if self.shield is not None:
            femm.mi_addmaterial(PLATE, 1, 1, 0, 0, self.shield.sigma * 1e-6,
                                0, 0, 1, 0, 0, 0, 0, 0)

    def _add_cables(self):
        r = self.cable_radius
        for k, (xc, zc, cur) in enumerate(
                zip(self.cables.xc, self.cables.zc, self.cables.I)):
            name = f"phase{k}"
            femm.mi_addcircprop(name, complex(cur), 1)      # 1 = series
            femm.mi_drawarc(xc - r, zc, xc + r, zc, 180, self.arc_seg)
            femm.mi_drawarc(xc + r, zc, xc - r, zc, 180, self.arc_seg)
            femm.mi_addblocklabel(xc, zc)
            femm.mi_selectlabel(xc, zc)
            # material, automesh, meshsize, incircuit, magdir, group, turns
            femm.mi_setblockprop(CABLE, 0, self.cable_mesh, name, 0, 1, 1)
            femm.mi_clearselected()

    def _add_shield(self):
        s = self.shield
        t2 = s.thickness / 2.0
        x0 = s.center_x - s.width / 2.0

        # Outer outline once, then the internal dividers, so no segment is drawn
        # twice (FEMM would otherwise stack coincident segments).
        femm.mi_drawrectangle(x0, -t2, x0 + s.width, t2)
        for k in range(1, s.n_plates):
            xk = x0 + k * s.plate_width
            femm.mi_drawline(xk, -t2, xk, t2)

        for name in s.circuit_names():
            femm.mi_addcircprop(name, 0.0, 0)   # 0 A net, 0 = parallel-connected

        for k in range(s.n_plates):
            xa, xb = s.plate_bounds(k)
            xm = 0.5 * (xa + xb)
            femm.mi_addblocklabel(xm, 0.0)
            femm.mi_selectlabel(xm, 0.0)
            femm.mi_setblockprop(PLATE, 0, self.plate_mesh,
                                 s.circuit_of(k), 0, 2, 1)
            femm.mi_clearselected()

    def _add_refine_box(self):
        if self.refine_box is None:
            return
        x0, z0, x1, z1 = self.refine_box
        femm.mi_drawrectangle(x0, z0, x1, z1)
        femm.mi_addblocklabel(0.5 * (x0 + x1), 0.5 * (z0 + z1))
        femm.mi_selectlabel(0.5 * (x0 + x1), 0.5 * (z0 + z1))
        femm.mi_setblockprop(AIR, 0, self.air_mesh, NO_CIRCUIT, 0, 3, 0)
        femm.mi_clearselected()

    def _add_air_label(self):
        femm.mi_addblocklabel(0.0, 5.0)
        femm.mi_selectlabel(0.0, 5.0)
        femm.mi_setblockprop(AIR, 1, 0, NO_CIRCUIT, 0, 0, 0)
        femm.mi_clearselected()

    # ---------------------------------------------------------------- solve

    def solve(self):
        femm.mi_analyze(1)                     # 1 = hide the solver window
        femm.mi_loadsolution()
        # Patch-recovery smoothing of B. Without it, point samples return the
        # raw element-wise-constant curl of a linear A.
        femm.mo_smooth("on")
        self._solved = True
        return self

    def sample_B(self, points):
        """B (complex phasor, Tesla) at `points` (N,3) in project coordinates.

        Returns an (N,3) complex array with the same layout and phasor convention
        as shielding.field.cable_field, so shielding.field.field_norm applies
        unchanged. The y-component is identically zero by 2-D invariance.
        """
        if not self._solved:
            raise RuntimeError("call solve() before sample_B()")
        pts = np.asarray(points, float)
        B = np.zeros((pts.shape[0], 3), dtype=complex)
        for i in range(pts.shape[0]):
            bx, bz = femm.mo_getb(float(pts[i, 0]), float(pts[i, 2]))
            B[i, 0] = bx
            B[i, 2] = bz
        return B

    def circuit_currents(self):
        """Net current per shield circuit -- the guard on the zero-net-current
        constraint. Every entry must be ~0; a non-zero one means the constraint
        was not applied and the shielding result is meaningless."""
        if self.shield is None:
            return {}
        out = {}
        for name in self.shield.circuit_names():
            props = femm.mo_getcircuitproperties(name)
            out[name] = complex(props[0])
        return out

    # ---------------------------------------------------------------- close

    def close(self):
        try:
            femm.mo_close()
        except Exception:
            pass
        try:
            femm.mi_close()
        except Exception:
            pass
        femm.closefemm()

    def __enter__(self):
        return self.build().solve()

    def __exit__(self, *exc):
        self.close()
        return False


def femm_norm(cables, points, shield=None, **kw):
    """|B| in microtesla at `points` for one FEMM case; returns (norm, currents)."""
    from shielding.field import field_norm
    with Femm2D(cables, shield=shield, **kw) as model:
        B = model.sample_B(points)
        currents = model.circuit_currents()
    return field_norm(B) * 1e6, currents
