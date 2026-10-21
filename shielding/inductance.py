"""Filament self- and mutual-inductance for the wire-mesh circuit method.

Formulas follow Clairmont & Lordan, IEEE T-PWRD 14(4) 1999 (Eq. 2-5), which are
the same ones used by Sergeant et al. (COMPEL 2008) and De Wulf et al. (JMMM 2007).

Key physical simplification used throughout the shield mesh:
    For two *straight* filaments the mutual inductance is
        M = (mu0/4pi) * INT INT (dl_a . dl_b)/r
    so it is proportional to the dot product of the two filament directions.
    Two perpendicular straight filaments therefore have M = 0 exactly.  In a
    planar, axis-aligned rectangular mesh every pair is either parallel
    (non-zero, handled here) or perpendicular (zero).  This makes the whole
    inductance assembly reduce to the well-behaved parallel-filament case.
"""

import numpy as np

MU0 = 4.0e-7 * np.pi
_C = MU0 / (4.0 * np.pi)          # = 1e-7


def self_inductance(length, radius):
    """Self-inductance of a straight round filament (Clairmont-Lordan Eq. 3).

    L = 2e-7 * l * (ln(4l/D) - 1),  with geometric mean diameter D = 2*r*e^(-1/4).
    Equivalent to the classic L = (mu0 l/2pi)(ln(2l/r) - 3/4).
    """
    length = np.asarray(length, dtype=float)
    radius = np.asarray(radius, dtype=float)
    gmd = 2.0 * radius * np.exp(-0.25)
    return 2.0e-7 * length * (np.log(4.0 * length / gmd) - 1.0)


def _Q(s, d):
    """Primitive for the parallel-filament Neumann double integral.

    d(2)Q/ds(2) relation gives, for perpendicular separation d > 0,
        Q(s, d) = sqrt(s^2 + d^2) - s * asinh(s/d).
    The mutual inductance of two collinear-axis parallel segments is the
    four-corner alternating sum of Q over the longitudinal end differences.
    """
    return np.sqrt(s * s + d * d) - s * np.arcsinh(s / d)


def _Q0(s):
    """Collinear limit (d -> 0) of the corner primitive: -|s| ln|s|.

    The |s| and ln(2) pieces of the full limit cancel in the four-corner sum
    for disjoint / edge-sharing collinear segments, leaving -|s| ln|s|.
    """
    s = np.abs(s)
    out = np.zeros_like(s)
    nz = s > 0
    out[nz] = -s[nz] * np.log(s[nz])
    return out


def mutual_parallel(a1, a2, b1, b2, d):
    """Mutual inductance of two parallel filaments (same direction).

    a1,a2 : longitudinal coordinates of filament A endpoints (on the common axis)
    b1,b2 : longitudinal coordinates of filament B endpoints
    d     : perpendicular distance between the two filament lines (>= 0)

    Vectorised: inputs may be scalars or broadcastable arrays.  Sign convention
    assumes both filaments point the same way; multiply by -1 for anti-parallel.
    """
    a1, a2, b1, b2, d = np.broadcast_arrays(
        *[np.asarray(x, dtype=float) for x in (a1, a2, b1, b2, d)]
    )
    d_safe = np.where(d > 0, d, 1.0)

    def corner(s):
        full = _Q(s, d_safe)
        coll = _Q0(s)
        return np.where(d > 0, full, coll)

    I = corner(a2 - b2) - corner(a2 - b1) - corner(a1 - b2) + corner(a1 - b1)
    return _C * I


def mutual_neumann_general(a0, a1, b0, b1, n=48):
    """General mutual inductance of two arbitrary 3-D filaments by Gauss-Legendre
    quadrature of the Neumann double integral.  Used only to unit-test
    `mutual_parallel`; not used in production assembly.

    a0,a1 : 3-vectors, endpoints of filament A
    b0,b1 : 3-vectors, endpoints of filament B
    """
    a0, a1, b0, b1 = (np.asarray(v, float) for v in (a0, a1, b0, b1))
    xs, ws = np.polynomial.legendre.leggauss(n)   # nodes on [-1, 1]
    t = 0.5 * (xs + 1.0)                            # -> [0, 1]
    wt = 0.5 * ws
    da = a1 - a0
    db = b1 - b0
    pa = a0[None, :] + t[:, None] * da[None, :]     # (n,3)
    pb = b0[None, :] + t[:, None] * db[None, :]     # (n,3)
    diff = pa[:, None, :] - pb[None, :, :]          # (n,n,3)
    r = np.sqrt(np.sum(diff * diff, axis=2))
    integrand = 1.0 / r
    val = np.einsum("i,j,ij->", wt, wt, integrand)
    dot = np.dot(da, db)
    return _C * dot * val
