"""Geometry: source cables and the filament mesh of the plate shield.

Coordinate system (matches Sergeant et al., COMPEL 2008, Fig. 1):
    x : horizontal, transverse to the cables
    y : along the cables (treated as infinitely long)
    z : vertical; z = 0 is the ground surface. The shield lies in the z = 0
        plane; cables are buried at z = -depth.

Contact resistance between plates
---------------------------------
Rc == 0  -> the plates are perfectly welded: a single conforming mesh with
            shared nodes at every junction (a monolithic sheet).
Rc  > 0  -> each plate is meshed with its own nodes; neighbouring plates are
            connected by short "contact" branches of resistance Rc between
            coincident boundary nodes (the crosses in Sergeant Fig. 2).
"""

import numpy as np
from .inductance import self_inductance


class Cables:
    """Three infinitely long phase conductors parallel to y, balanced 3-phase."""

    def __init__(self, spacing=0.25, depth=1.5, current=900.0, seq=+1):
        self.xc = np.array([-spacing, 0.0, spacing])
        self.zc = np.array([-depth, -depth, -depth])
        ph = np.array([2 * np.pi / 3, 0.0, -2 * np.pi / 3]) * seq
        self.I = current * np.exp(1j * ph)     # phasor amplitudes (A)


class _Builder:
    """Accumulates deduplicated nodes and branches, then freezes into a Mesh."""

    def __init__(self):
        self._nodes = {}          # key -> id
        self.node_xyz = []        # id -> (x, y, z)
        self.b_from, self.b_to = [], []
        self.b_axis = []          # 0 = x, 1 = y, -1 = contact
        self.b_len = []
        self.b_Rextra = []        # extra series resistance (contact)

    def node(self, key, xyz):
        if key not in self._nodes:
            self._nodes[key] = len(self.node_xyz)
            self.node_xyz.append(xyz)
        return self._nodes[key]

    def branch(self, nf, nt, axis, length, r_extra=0.0):
        self.b_from.append(nf)
        self.b_to.append(nt)
        self.b_axis.append(axis)
        self.b_len.append(length)
        self.b_Rextra.append(r_extra)


class Mesh:
    """Frozen filament mesh with everything the solver and field code need."""

    def __init__(self, builder, radius, rho_R, contact_nom_len):
        b = builder
        self.node_xyz = np.asarray(b.node_xyz, dtype=float)
        self.n_nodes = len(self.node_xyz)
        self.branch_from = np.asarray(b.b_from)
        self.branch_to = np.asarray(b.b_to)
        self.branch_axis = np.asarray(b.b_axis)
        self.branch_len = np.asarray(b.b_len, dtype=float)
        self.n_branches = len(self.branch_from)
        self.radius = radius
        self.rho_R = rho_R

        p_from = self.node_xyz[self.branch_from]
        p_to = self.node_xyz[self.branch_to]
        mid = 0.5 * (p_from + p_to)
        # transverse (in-plane) coordinate of each filament line:
        #   x-branch (axis 0): transverse = y ; y-branch (axis 1): transverse = x
        self.branch_transverse = np.where(self.branch_axis == 1, mid[:, 0], mid[:, 1])
        # longitudinal endpoint coordinates on the branch's own axis
        ax_is_y = self.branch_axis == 1
        self.s_lo = np.where(ax_is_y, np.minimum(p_from[:, 1], p_to[:, 1]),
                             np.minimum(p_from[:, 0], p_to[:, 0]))
        self.s_hi = np.where(ax_is_y, np.maximum(p_from[:, 1], p_to[:, 1]),
                             np.maximum(p_from[:, 0], p_to[:, 0]))

        # resistance: base sheet resistance + contact extra
        base_R = rho_R * self.branch_len
        self.R = base_R + np.asarray(b.b_Rextra, dtype=float)

        # self inductance: mesh branches from their length; contact branches
        # (axis -1, ~zero length) from a small nominal length so L is finite.
        eff_len = np.where(self.branch_axis == -1, contact_nom_len, self.branch_len)
        self.L_self = self_inductance(np.maximum(eff_len, 1e-6), radius)

    def segment_endpoints(self):
        return self.node_xyz[self.branch_from], self.node_xyz[self.branch_to]


def _equiv_wire(W, L, thickness, n_ywires, n_xwires, sigma):
    """Clairmont-Lordan equivalent wire radius and resistance-per-length via
    volume conservation over the whole shield area (W x L x thickness)."""
    total_wire_len = n_ywires * L + n_xwires * W
    area = W * L * thickness
    radius = np.sqrt(area / (np.pi * total_wire_len))
    rho_R = 1.0 / (sigma * np.pi * radius ** 2)
    return radius, rho_R


def build_shield(config="parallel", W=4.0, L=4.0, n_plates=8,
                 n_across=3, n_along=16, sigma=5.8e7, thickness=3e-3,
                 Rc=0.0, center=(0.0, 0.0)):
    """Assemble the standard multi-plate shield.

    config="parallel"   : plate long edge parallel to cables (y); n_plates strips
                          side by side across x.  mesh nx=n_plates*n_across, ny=n_along
    config="transversal": plate long edge orthogonal to cables; strips stacked
                          along y.  mesh nx=n_along, ny=n_plates*n_across

    n_across : filaments across one plate's short dimension (Sergeant n_b)
    n_along  : filaments along the shield long dimension (Sergeant n_l)
    """
    cx, cy = center
    xmin, ymin = cx - W / 2, cy - L / 2

    if config == "parallel":
        nx, ny = n_plates * n_across, n_along
        # plate index of a cell column i -> i // n_across ; junctions across x
        plate_of_col = lambda i: i // n_across
        split_axis = "x"
    elif config == "transversal":
        nx, ny = n_along, n_plates * n_across
        plate_of_row = lambda j: j // n_across
        split_axis = "y"
    else:
        raise ValueError(config)

    dx, dy = W / nx, L / ny
    xs = xmin + dx * np.arange(nx + 1)
    ys = ymin + dy * np.arange(ny + 1)
    radius, rho_R = _equiv_wire(W, L, thickness, nx + 1, ny + 1, sigma)
    contact_nom_len = 0.5 * (dx + dy) * 0.25

    bld = _Builder()

    def which_plate(i, j):
        if config == "parallel":
            return min(i, nx - 1) // n_across if i < nx else (nx - 1) // n_across
        else:
            return min(j, ny - 1) // n_across if j < ny else (ny - 1) // n_across

    welded = (Rc == 0.0)

    # Node key: interior/edge nodes are shared globally when welded; when not
    # welded, nodes on a junction line get a per-plate suffix so the plates are
    # electrically separate there.
    if config == "parallel":
        junction_cols = set(range(n_across, nx, n_across))   # interior x-junctions

        def nkey(i, j, side_plate):
            if (not welded) and (i in junction_cols or i - 0 in ()):
                return (i, j, side_plate)
            return (i, j)
    else:
        junction_rows = set(range(n_across, ny, n_across))

        def nkey(i, j, side_plate):
            if (not welded) and (j in junction_rows):
                return (i, j, side_plate)
            return (i, j)

    def add_node(i, j, plate):
        return bld.node(nkey(i, j, plate), (xs[i], ys[j], 0.0))

    # Build cell by cell so each plate owns a coherent set of nodes.
    for i in range(nx):
        for j in range(ny):
            if config == "parallel":
                pL, pR = i // n_across, i // n_across   # cell belongs to one plate
                plate = i // n_across
            else:
                plate = j // n_across
            # nodes of this cell (belong to this plate's copy on junction lines)
            n00 = add_node(i, j, plate)
            n10 = add_node(i + 1, j, plate)
            n01 = add_node(i, j + 1, plate)
            # x-branch (i,j)->(i+1,j) along bottom edge; add top row later
            bld.branch(n00, n10, 0, dx)
            # y-branch (i,j)->(i,j+1) along left edge
            bld.branch(n00, n01, 1, dy)
        # top row x-branches and right column handled by neighbours / closure

    # close the far (top) x-branches for j = ny
    for i in range(nx):
        if config == "parallel":
            plate = i // n_across
        else:
            plate = (ny - 1) // n_across
        nA = add_node(i, ny, plate)
        nB = add_node(i + 1, ny, plate)
        bld.branch(nA, nB, 0, dx)
    # close the far (right) y-branches for i = nx
    for j in range(ny):
        if config == "parallel":
            plate = (nx - 1) // n_across
        else:
            plate = j // n_across
        nA = add_node(nx, j, plate)
        nB = add_node(nx, j + 1, plate)
        bld.branch(nA, nB, 1, dy)

    # contact branches for the unwelded case
    if not welded:
        if config == "parallel":
            for i in junction_cols:
                for j in range(ny + 1):
                    left = bld.node((i, j, i // n_across - 1), (xs[i], ys[j], 0.0))
                    right = bld.node((i, j, i // n_across), (xs[i], ys[j], 0.0))
                    if left != right:
                        bld.branch(left, right, -1, 0.0, r_extra=Rc)
        else:
            for j in junction_rows:
                for i in range(nx + 1):
                    lo = bld.node((i, j, j // n_across - 1), (xs[i], ys[j], 0.0))
                    hi = bld.node((i, j, j // n_across), (xs[i], ys[j], 0.0))
                    if lo != hi:
                        bld.branch(lo, hi, -1, 0.0, r_extra=Rc)

    return Mesh(bld, radius, rho_R, contact_nom_len)
