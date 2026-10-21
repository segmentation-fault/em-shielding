import os, sys, numpy as np
os.environ["ELMER_WSL_DISTRO"]="Ubuntu-24.04"; os.environ["ELMER_WSL_HOME"]="$HOME/elmer"
sys.path.insert(0, r"D:\devel\shielding")
from fem.elmer3d import _wsl_run, _read_saveline, analytic_line_B, Cables3D
from shielding.field import field_norm
import coil3_geom
wd=r"C:\Users\Admin\AppData\Local\Temp\claude\D--devel-shielding\f1d730b9-e9f1-4b7a-b45f-587b04de3e58\scratchpad\wc3"
os.makedirs(wd, exist_ok=True)
xcs=(-0.25,0.0,0.25)
ids=coil3_geom.build(os.path.join(wd,"model.msh"), xcs=xcs)
print(_wsl_run("Ubuntu-24.04","$HOME/elmer",wd,"$HOME/elmer/bin/ElmerGrid 14 2 model.msh -out mesh -autoclean").stdout[-80:])
w=2*np.pi*50
I0=900.0; ph=[2*np.pi/3,0.0,-2*np.pi/3]
S=[(I0*np.cos(p), I0*np.sin(p)) for p in ph]

# circuit definitions: 3 independent current-source circuits
d=["$ Circuits = 3\n"]
for n in (1,2,3):
    d.append(f"""$ C.{n}.perm = zeros(4)
$ C.{n}.perm(0) = 0
$ C.{n}.perm(2) = 1
$ C.{n}.perm(3) = 2
$ C.{n}.perm(1) = 3
$ C.{n}.variables = 4
$ C.{n}.A = zeros(4,4)
$ C.{n}.B = zeros(4,4)
$ C.{n}.Mre = zeros(4,4)
$ C.{n}.Mim = zeros(4,4)
$ C.{n}.name.1 = "i_S"
$ C.{n}.name.2 = "v_S"
$ C.{n}.name.3 = "i_component({n})"
$ C.{n}.name.4 = "v_component({n})"
$ C.{n}.B(0,0) = 1
$ C.{n}.source.1 = "S"
$ C.{n}.B(1,1) = 1
$ C.{n}.B(1,3) = -1
$ C.{n}.B(2,0) = 1
$ C.{n}.B(2,2) = 1
""")
open(os.path.join(wd,"circuits.definitions"),"w").write("".join(d))

def bodies():
    s=['Body 1\n  Name = air\n  Target Bodies(1) = 1\n  Equation = 1\n  Material = 1\n  Initial Condition = 1\nEnd\n']
    for k in range(3):
        s.append(f'Body {2+k}\n  Name = coil{k}\n  Target Bodies(1) = {2+k}\n  Equation = 2\n'
                 f'  Material = 2\n  Initial Condition = 1\n  Body Force = {1+k}\n'
                 f'  Alpha reference (3) = Real 1 0 0\n  Beta reference (3) = Real 0 0 1\nEnd\n')
    return "".join(s)
def comps():
    s=[]
    for k in range(3):
        s.append(f'Component {1+k}\n  Name = String coil{k}\n  Body = Integer {2+k}\n'
                 f'  Coil Type = String stranded\n  Number of Turns = Real 1\n'
                 f'  Electrode Boundaries(2) = Integer {ids[f"gamma0_{k}"]} {ids[f"gamma1_{k}"]}\n'
                 f'  Circuit Equation Voltage Factor = Real 1\nEnd\n')
    return "".join(s)
def bforces():
    s=[]
    for k in range(3):
        s.append(f'Body Force {1+k}\n  Name = "Circuit{k}"\n'
                 f'  S Re = Real {S[k][0]:.8e}\n  S Im = Real {S[k][1]:.8e}\nEnd\n')
    return "".join(s)
def dirbcs():
    s=[f'Boundary Condition 1\n  Target Boundaries(1) = {ids["outer"]}\n  A re {{e}} = Real 0\n  A im {{e}} = Real 0\nEnd\n']
    bc=2
    for k in range(3):
        s.append(f'Boundary Condition {bc}\n  Target Boundaries(1) = {ids[f"alpha0_{k}"]}\n  Body {2+k}: Alpha = Real 0\nEnd\n'); bc+=1
        s.append(f'Boundary Condition {bc}\n  Target Boundaries(1) = {ids[f"alpha1_{k}"]}\n  Body {2+k}: Alpha = Real 1\nEnd\n'); bc+=1
        s.append(f'Boundary Condition {bc}\n  Target Boundaries(1) = {ids[f"beta0_{k}"]}\n  Body {2+k}: Beta = Real 0\nEnd\n'); bc+=1
        s.append(f'Boundary Condition {bc}\n  Target Boundaries(1) = {ids[f"beta1_{k}"]}\n  Body {2+k}: Beta = Real 1\nEnd\n'); bc+=1
        s.append(f'Boundary Condition {bc}\n  Target Boundaries(1) = {ids[f"gamma0_{k}"]}\n  W = Real 0\nEnd\n'); bc+=1
        s.append(f'Boundary Condition {bc}\n  Target Boundaries(1) = {ids[f"gamma1_{k}"]}\n  W = Real 1\nEnd\n'); bc+=1
    return "".join(s)

def solver(n, txt): return f"Solver {n}\n{txt}End\n"
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
  Electric Conductivity = 0
  Relative Permeability = 1
End
Material 2
  Electric Conductivity = 5.8e7
  Relative Permeability = 1
End
{bodies()}{comps()}{bforces()}
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
  Linear System Solver = Iterative
  Linear System Iterative Method = BicgstabL
  Linear System preconditioning = Circuit
  Linear System Convergence Tolerance = 1.e-7
  Linear System Max Iterations = 5000
  BicgStabL Polynomial Degree = 4
  Linear System Abort not Converged = False
  Linear System Residual Output = 500
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
{dirbcs()}"""
open(os.path.join(wd,"case.sif"),"w").write(sif)
open(os.path.join(wd,"ELMERSOLVER_STARTINFO"),"w").write("case.sif\n")
out=_wsl_run("Ubuntu-24.04","$HOME/elmer",wd,"$HOME/elmer/bin/ElmerSolver case.sif")
ok="ALL DONE" in out.stdout
print("ALL DONE:", ok)
if not ok:
    print(out.stdout[-2500:])
else:
    open(os.path.join(wd,"solver.log"),"w").write(out.stdout)
    for ln in out.stdout.splitlines():
        if any(w in ln.lower() for w in ["i_component","current in","coil0","coil1","coil2"]) and "unlisted" not in ln.lower(): print("CIRC:", ln.strip()[:120])
    X=np.linspace(0,4,21); pts=np.column_stack([X,0*X,0*X+1])
    el=field_norm(_read_saveline(os.path.join(wd,"bline.dat"),pts))*1e6
    an=field_norm(analytic_line_B(Cables3D(),pts))*1e6
    for i in range(0,21,2): print(f"x{X[i]:.1f} elmer {el[i]:8.3f} analytic {an[i]:7.3f} rel {100*abs(el[i]-an[i])/an[i]:6.1f}%")
    rel=np.abs(el-an)/an; print(f"J0 RMS {100*np.sqrt(np.mean(rel**2)):.2f}%")
