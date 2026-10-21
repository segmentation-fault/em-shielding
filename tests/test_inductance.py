"""Unit tests for the inductance formulas (run with the project venv)."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shielding.inductance import (
    self_inductance, mutual_parallel, mutual_neumann_general, MU0,
)


def test_self_inductance_matches_classic():
    l, r = 1.0, 1e-3
    L = self_inductance(l, r)
    classic = (MU0 * l / (2 * np.pi)) * (np.log(2 * l / r) - 0.75)
    assert abs(L - classic) / classic < 1e-12, (L, classic)


def test_long_parallel_limit():
    # Two long parallel filaments length l, separation d, l >> d.
    l, d = 100.0, 0.01
    M = mutual_parallel(0.0, l, 0.0, l, d)
    approx = (MU0 * l / (2 * np.pi)) * (np.log(2 * l / d) - 1.0)
    assert abs(M - approx) / approx < 1e-3, (M, approx)


def test_parallel_vs_neumann():
    # Offset parallel filaments, generic geometry: closed form vs quadrature.
    a0, a1 = np.array([0, 0, 0.0]), np.array([1.0, 0, 0])      # along x, [0,1]
    b0, b1 = np.array([0.3, 0.25, 0.0]), np.array([1.4, 0.25, 0])  # parallel, offset
    M_num = mutual_neumann_general(a0, a1, b0, b1, n=200)
    # longitudinal coords along x; perpendicular distance = 0.25 (y), z equal
    M_cf = mutual_parallel(a0[0], a1[0], b0[0], b1[0], 0.25)
    assert abs(M_num - M_cf) / abs(M_num) < 1e-3, (M_num, M_cf)


def test_collinear_positive_and_finite():
    # Two disjoint collinear segments, same direction -> positive, finite.
    M = mutual_parallel(0.0, 1.0, 2.0, 3.0, 0.0)
    a0, a1 = np.array([0, 0, 0.0]), np.array([1.0, 0, 0])
    b0, b1 = np.array([2.0, 0, 0.0]), np.array([3.0, 0, 0])
    M_num = mutual_neumann_general(a0, a1, b0, b1, n=400)
    assert M > 0
    assert abs(M - M_num) / M_num < 5e-3, (M, M_num)


def test_perpendicular_zero():
    a0, a1 = np.array([0, 0, 0.0]), np.array([1.0, 0, 0])
    b0, b1 = np.array([0.5, 0, 0.0]), np.array([0.5, 1.0, 0])  # perpendicular
    M_num = mutual_neumann_general(a0, a1, b0, b1, n=200)
    assert abs(M_num) < 1e-12, M_num


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("all inductance tests passed")
