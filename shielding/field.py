"""Magnetic flux density: infinite-line source cables and finite filament segments."""

import numpy as np
from .inductance import MU0


def cable_field(cables, points):
    """B (complex phasor, Tesla) at `points` (N,3) from the infinite-line cables.

    For an infinite line current I along +y at (xc, zc):
        B_x =  mu0 I /(2pi r^2) (pz - zc)
        B_z = -mu0 I /(2pi r^2) (px - xc),   r^2 = (px-xc)^2 + (pz-zc)^2
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


def A_y_source(cables, x, z=0.0):
    """y-component of the source vector potential at (x, z) (per-cable ln form).

    A_y = -(mu0/2pi) sum_k I_k ln(r_k).  The arbitrary additive constant cancels
    in the mesh loop equations, so any length reference for r is admissible.
    """
    x = np.asarray(x, float)
    Ay = np.zeros(x.shape, dtype=complex)
    for xc, zc, I in zip(cables.xc, cables.zc, cables.I):
        r = np.sqrt((x - xc) ** 2 + (z - zc) ** 2)
        Ay += -(MU0 / (2 * np.pi)) * I * np.log(r)
    return Ay


def biot_savart_segments(p_from, p_to, I_branch, points):
    """B (complex, Tesla) at `points` (N,3) from straight segments carrying I_branch.

    p_from, p_to : (M,3) segment endpoints; I_branch : (M,) complex currents.
    Closed-form finite-wire Biot-Savart, vectorised over M segments x N points.
    """
    p_from = np.asarray(p_from, float)
    p_to = np.asarray(p_to, float)
    Q = np.asarray(points, float)
    Lvec = p_to - p_from                      # (M,3)
    ell = np.linalg.norm(Lvec, axis=1)        # (M,)
    good = ell > 1e-12
    e = np.zeros_like(Lvec)
    e[good] = Lvec[good] / ell[good, None]

    B = np.zeros((Q.shape[0], 3), dtype=complex)
    # loop over points (N is small: profiles of a few dozen points)
    for q in range(Q.shape[0]):
        r1 = Q[q][None, :] - p_from            # (M,3)
        t = np.einsum("mk,mk->m", r1, e)       # projection of Q onto line
        r_perp = r1 - t[:, None] * e
        d = np.linalg.norm(r_perp, axis=1)     # (M,)
        s_a = -t
        s_b = ell - t
        with np.errstate(divide="ignore", invalid="ignore"):
            bracket = (s_b / np.sqrt(s_b ** 2 + d ** 2)
                       - s_a / np.sqrt(s_a ** 2 + d ** 2))
            d_hat = np.where(d[:, None] > 1e-12, r_perp / np.where(d[:, None] > 0, d[:, None], 1), 0.0)
            dir_vec = np.cross(e, d_hat)       # (M,3), unit
            mag = MU0 * I_branch / (4 * np.pi) * np.where(d > 1e-12, bracket / np.where(d > 0, d, 1), 0.0)
        contrib = mag[:, None] * dir_vec       # (M,3) complex
        B[q] = contrib.sum(axis=0)
    return B


def field_norm(B):
    """Resultant norm of a complex phasor field: sqrt(|Bx|^2+|By|^2+|Bz|^2)."""
    return np.sqrt(np.sum(np.abs(B) ** 2, axis=1))
