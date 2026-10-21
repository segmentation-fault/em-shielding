"""Shared setup for figure scripts: path, style, standard geometry."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shielding.geometry import Cables, build_shield          # noqa: E402
from shielding.analysis import (profile_points, unshielded_norm,  # noqa: E402
                                shielded_norm)

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Sergeant Table I ("Simulation" column)
SPACING = 0.25      # distance between cables (m)
DEPTH = 1.5         # cable depth below ground (m)
CURRENT = 900.0     # phase current amplitude (A)
ZM = 1.0            # evaluation height above ground (m)

plt.rcParams.update({
    "figure.figsize": (7.2, 5.0),
    "figure.dpi": 130,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "lines.linewidth": 1.8,
})


# Series colours for the FEM-comparison figures. Okabe-Ito blue/vermillion:
# a colour-vision-safe adjacent pair (worst-case protan dE 21.9, normal 31.2).
# Method is encoded by linestyle and case by colour, so two hues suffice and
# nothing depends on colour alone.
C_WELDED = "#0072B2"        # welded shield / Rc = 0
C_ISOLATED = "#D55E00"      # isolated plates / large Rc
C_THIRD = "#009E73"         # third categorical slot (worst pair deutan dE 11.0)
C_REF = "#444444"           # unshielded reference
LS_FEMM = "--"              # FEM result
LS_CIRCUIT = "-"            # circuit method


def length_ramp(n):
    """Sequential single-hue ramp, light -> dark, for the plate-length sweep
    (length is a magnitude, so it gets a ramp rather than categorical hues)."""
    return [plt.get_cmap("Blues")(v) for v in np.linspace(0.40, 0.95, n)]


def standard_cables():
    return Cables(spacing=SPACING, depth=DEPTH, current=CURRENT)


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches="tight")
    print("wrote", path)
