import os, sys, numpy as np
os.environ["ELMER_WSL_DISTRO"]="Ubuntu-24.04"; os.environ["ELMER_WSL_HOME"]="$HOME/elmer"
sys.path.insert(0, r"D:\devel\shielding")
from fem.elmer3d import _wsl_run, _read_saveline, MU0
import coil_geom
wd=r"C:\Users\Admin\AppData\Local\Temp\claude\D--devel-shielding\f1d730b9-e9f1-4b7a-b45f-587b04de3e58\scratchpad\wcoil"
ids=coil_geom.build(os.path.join(wd,"model.msh"))
# convert mesh
print(_wsl_run("Ubuntu-24.04","$HOME/elmer",wd,"$HOME/elmer/bin/ElmerGrid 14 2 model.msh -out mesh -autoclean").stdout[-200:])

Ire, Iim = 900.0, 0.0     # target coil current (peak)
w = 2*np.pi*50

defs = """$ Circuits = 1
$ C.1.perm = zeros(4)
$ C.1.perm(0) = 0
$ C.1.perm(2) = 1
$ C.1.perm(3) = 2
$ C.1.perm(1) = 3
$ C.1.variables = 4
$ C.1.A = zeros(4,4)
$ C.1.B = zeros(4,4)
$ C.1.Mre = zeros(4,4)
$ C.1.Mim = zeros(4,4)
$ C.1.name.1 = "i_S"
$ C.1.name.2 = "v_S"
$ C.1.name.3 = "i_component(1)"
$ C.1.name.4 = "v_component(1)"
$ C.1.B(0,0) = 1
$ C.1.source.1 = "S"
$ C.1.B(1,1) = 1
$ C.1.B(1,3) = -1
$ C.1.B(2,0) = 1
$ C.1.B(2,2) = 1
"""
open(os.path.join(wd,"circuits.definitions"),"w").write(defs)

sif=f"""Check Keywords "Warn"
INCLUDE circuits.definitions
Header
  Mesh DB "." "mesh"
End
Simulation
  Max Output Level = 4
  Coordinate System = Cartesian 3D
  Simulation Type = Steady
  Steady State Max Iterations = 1
  Angular Frequency = {w}
End
Initial Condition 1
  A re = Real 0
  A im = Real 0
  A re {{e}} = Real 0
  A im {{e}} = Real 0
End
Material 1
  Name = air
  Electric Conductivity = 0
  Relative Permeability = 1
End
Material 2
  Name = metal
  Electric Conductivity = 5.8e7
  Relative Permeability = 1
End
Body 1
  Name = air
  Target Bodies(1) = 1
  Equation = 1
  Material = 1
  Initial Condition = 1
End
Body 2
  Name = coil
  Target Bodies(1) = 2
  Equation = 2
  Material = 2
  Initial Condition = 1
  Body Force = 1
  Alpha reference (3) = Real 1 0 0
  Beta reference (3) = Real 0 0 1
End
Component 1
  Name = String coil
  Body = Integer 2
  Coil Type = String stranded
  Number of Turns = Real 1
  Electrode Boundaries(2) = Integer {ids['gamma0']} {ids['gamma1']}
  Circuit Equation Voltage Factor = Real 1
End
Body Force 1
  Name = "Circuit"
  S Re = Real {Ire}
  S Im = Real {Iim}
End
Solver 1
  Exec Solver = Before all
  Procedure = "DirectionSolver" "DirectionSolver"
  Equation = Direction Alpha
  Variable = Alpha
  Linear System Solver = Iterative
  Linear System Iterative Method = BiCGStab
  Linear System Max Iterations = 5000
  Linear System Convergence Tolerance = 1.0e-9
  Linear System Preconditioning = ILU1
  Linear System Residual Output = 0
End
Solver 2
  Exec Solver = Before all
  Procedure = "DirectionSolver" "DirectionSolver"
  Equation = Direction Beta
  Variable = Beta
  Linear System Solver = Iterative
  Linear System Iterative Method = BiCGStab
  Linear System Max Iterations = 5000
  Linear System Convergence Tolerance = 1.0e-9
  Linear System Preconditioning = ILU1
  Linear System Residual Output = 0
End
Solver 3
  Exec Solver = Before All
  Equation = Transformation matrix
  Procedure = "CoordinateTransform" "RotMSolver"
  Variable = -nooutput Dummy
  Optimize Bandwidth = False
  Exported Variable 1 = -nooutput RotM E[RotM E:9]
  Exported Variable 2 = -nooutput Alpha Vector E[Alpha Vector E:3]
  Exported Variable 3 = -nooutput Beta Vector E[Beta Vector E:3]
  Exported Variable 4 = -nooutput Gamma Vector E[Gamma Vector E:3]
  Discontinuous Galerkin = True
End
Solver 4
  Exec Solver = Before All
  Procedure = "WPotentialSolver" "Wsolve"
  Equation = "Wire direction"
  Variable = W
  Linear System Solver = Iterative
  Linear System Iterative Method = CG
  Linear System Max Iterations = 10000
  Linear System Convergence Tolerance = 1.0e-9
  Linear System Preconditioning = ILU1
  Linear System Residual Output = 0
End
Solver 5
  Exec Solver = Always
  Equation = Circuits
  Variable = X
  No Matrix = Logical True
  Procedure = "CircuitsAndDynamics" "CircuitsAndDynamicsHarmonic"
End
Solver 6
  Exec Solver = Always
  Equation = "MGDynamics"
  Variable = "A[A re:1 A im:1]"
  Procedure = "MagnetoDynamics" "WhitneyAVHarmonicSolver"
  Angular Frequency = {w}
  Export Lagrange Multiplier = Logical True
  Linear System Symmetric = Logical True
  Linear System Complex = Logical True
  Linear System Solver = Direct
  Linear System Direct Method = MUMPS
  Steady State Convergence Tolerance = 1e-6
End
Solver 7
  Exec Solver = Always
  Equation = "MGDynamicsCalc"
  Procedure = "MagnetoDynamics" "MagnetoDynamicsCalcFields"
  Potential Variable = String "A"
  Angular Frequency = {w}
  Calculate Magnetic Flux Density = Logical True
  Linear System Solver = Iterative
  Linear System Iterative Method = CG
  Linear System Preconditioning = None
  Linear System Max Iterations = 5000
  Linear System Convergence Tolerance = 1.0e-8
  Linear System Residual Output = 0
End
Solver 8
  Exec Solver = Always
  Equation = Circuits Output
  Procedure = "CircuitsAndDynamics" "CircuitsOutput"
End
Solver 9
  Exec Solver = After All
  Equation = "SaveLine"
  Procedure = "SaveData" "SaveLine"
  Filename = "bline.dat"
  Polyline Coordinates(2,3) = Real 0.0 0.0 1.0  4.0 0.0 1.0
  Polyline Divisions(1) = Integer 20
End
Equation 1
  Active Solvers(2) = 6 7
End
Equation 2
  Active Solvers(7) = 1 2 3 4 5 6 7
End
Boundary Condition 1
  Target Boundaries(1) = {ids['outer']}
  A re {{e}} = Real 0
  A im {{e}} = Real 0
End
Boundary Condition 2
  Target Boundaries(1) = {ids['alpha0']}
  Body 2: Alpha = Real 0
End
Boundary Condition 3
  Target Boundaries(1) = {ids['alpha1']}
  Body 2: Alpha = Real 1
End
Boundary Condition 4
  Target Boundaries(1) = {ids['beta0']}
  Body 2: Beta = Real 0
End
Boundary Condition 5
  Target Boundaries(1) = {ids['beta1']}
  Body 2: Beta = Real 1
End
Boundary Condition 6
  Target Boundaries(1) = {ids['gamma0']}
  W = Real 0
End
Boundary Condition 7
  Target Boundaries(1) = {ids['gamma1']}
  W = Real 1
End
"""
open(os.path.join(wd,"case.sif"),"w").write(sif)
open(os.path.join(wd,"ELMERSOLVER_STARTINFO"),"w").write("case.sif\n")
out=_wsl_run("Ubuntu-24.04","$HOME/elmer",wd,"$HOME/elmer/bin/ElmerSolver case.sif")
ok = "ALL DONE" in out.stdout
print("ALL DONE:", ok)
if not ok:
    print(out.stdout[-2000:])
else:
    # coil current from CircuitsOutput
    import re
    for ln in out.stdout.splitlines():
        if "i_component" in ln.lower() or "i_s" in ln.lower(): print("CIRC:", ln.strip())
    X=np.linspace(0,4,21); pts=np.column_stack([X,0*X,0*X+1])
    el=(np.abs(_read_saveline(os.path.join(wd,"bline.dat"),pts))**2).sum(1)**0.5*1e6
    line=MU0/(2*np.pi)*900/np.sqrt(X**2+2.5**2)*1e6
    for i in range(0,21,3): print(f"x{X[i]:.1f} elmer {el[i]:8.3f} line {line[i]:7.3f} ratio {el[i]/line[i]:.3f}")
