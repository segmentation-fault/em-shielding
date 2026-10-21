"""Circuit method: assemble and solve the passive filament network.

Solves the branch currents induced in the shield mesh by the 3-phase cables,
following Sergeant et al. (COMPEL 2008), Eq. 2-4:

    [ R + jwL    -T^T ] [ I_t ]   [ -V_i ]
    [   T          0  ] [ V_n ] = [   0  ]

with T the reduced node-branch incidence matrix (KCL), V_i the source-induced
EMF per branch, and V_n the node potentials.
"""

import numpy as np
from .inductance import mutual_parallel
from .field import A_y_source


def assemble_inductance(mesh):
    """Dense mutual+self inductance matrix (n_branches x n_branches).

    Only parallel filaments couple; perpendicular straight filaments have zero
    mutual inductance, so the matrix is block-diagonal in {x-branches, y-branches}
    with contact branches carrying self inductance only.
    """
    n = mesh.n_branches
    Lm = np.zeros((n, n))
    np.fill_diagonal(Lm, mesh.L_self)
    for ax in (0, 1):
        idx = np.where(mesh.branch_axis == ax)[0]
        if idx.size == 0:
            continue
        s1 = mesh.s_lo[idx]
        s2 = mesh.s_hi[idx]
        t = mesh.branch_transverse[idx]
        d = np.abs(t[:, None] - t[None, :])
        M = mutual_parallel(s1[:, None], s2[:, None],
                            s1[None, :], s2[None, :], d)
        M[np.arange(idx.size), np.arange(idx.size)] = mesh.L_self[idx]
        Lm[np.ix_(idx, idx)] = M
    return Lm


def incidence(mesh):
    """Reduced incidence matrix T (n_nodes-1back x n_branches), reference node 0 dropped.
    T[node, br] = +1 if br starts at node, -1 if it ends there."""
    n_nodes = mesh.n_nodes
    n_br = mesh.n_branches
    T = np.zeros((n_nodes, n_br))
    T[mesh.branch_from, np.arange(n_br)] += 1.0
    T[mesh.branch_to, np.arange(n_br)] -= 1.0
    return T[1:, :]                      # drop reference node


def source_emf(mesh, cables, omega):
    """V_i per branch = -jw * A_source . dl.  Non-zero only on y-branches."""
    p_from, p_to = mesh.segment_endpoints()
    dl = p_to - p_from                    # (M,3)
    Vi = np.zeros(mesh.n_branches, dtype=complex)
    ax_y = mesh.branch_axis == 1
    x_mid = 0.5 * (p_from[ax_y, 0] + p_to[ax_y, 0])
    Ay = A_y_source(cables, x_mid, z=0.0)
    # From Faraday+Ohm around a branch: (R+jwL)I - Vt = -jw INT A_src.dl,
    # and Sergeant's Z It + Vi = Vt gives Vi = +jw INT A_src.dl.
    Vi[ax_y] = 1j * omega * Ay * dl[ax_y, 1]
    return Vi


def solve_currents(mesh, cables, freq=50.0):
    """Return the complex branch currents I_t induced in the shield."""
    omega = 2 * np.pi * freq
    Lm = assemble_inductance(mesh)
    Z = np.diag(mesh.R).astype(complex) + 1j * omega * Lm
    T = incidence(mesh)
    Vi = source_emf(mesh, cables, omega)

    nb = mesh.n_branches
    nn = T.shape[0]
    A = np.zeros((nb + nn, nb + nn), dtype=complex)
    A[:nb, :nb] = Z
    A[:nb, nb:] = -T.T
    A[nb:, :nb] = T
    rhs = np.zeros(nb + nn, dtype=complex)
    rhs[:nb] = -Vi
    sol = np.linalg.solve(A, rhs)
    return sol[:nb]
