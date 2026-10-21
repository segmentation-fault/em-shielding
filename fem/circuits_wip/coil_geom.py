"""Single straight stranded-coil mesh: one cable bar (internal) in an air box,
with the 6 cable faces tagged for the DirectionSolver/WPotential chain:
  gamma0 (-y end), gamma1 (+y end)   -- current electrodes / W potential ends
  alpha0 (-x), alpha1 (+x)           -- cross-section alpha direction
  beta0  (-z), beta1  (+z)           -- cross-section beta direction
plus the air outer boundary (flux parallel).
Bodies: air=1, coil=2.
"""
import sys, gmsh, numpy as np

def build(msh, box=6.0, cl=3.0, cs=0.3, depth=1.5, lc_far=1.0, lc_coil=0.12,
          lc_sample=0.15, sample_x=4.2):
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("coil"); occ = gmsh.model.occ
    eps = 1e-3
    b = occ.addBox(-box, -box, -box, 2*box, 2*box, 2*box)
    c = occ.addBox(-cs/2, -cl, -depth-cs/2, cs, 2*cl, cs)
    occ.fragment([(3,b)], [(3,c)]); occ.synchronize()
    air=coil=None
    for (_,t) in gmsh.model.getEntities(3):
        x0,y0,z0,x1,y1,z1 = gmsh.model.getBoundingBox(3,t)
        if (x1-x0)<1 and abs((z0+z1)/2+depth)<0.1: coil=t
        else: air=t
    # faces
    tags={"outer":[], "gamma0":[], "gamma1":[], "alpha0":[], "alpha1":[],
          "beta0":[], "beta1":[]}
    for (_,t) in gmsh.model.getEntities(2):
        x0,y0,z0,x1,y1,z1 = gmsh.model.getBoundingBox(2,t)
        cx,cy,cz=(x0+x1)/2,(y0+y1)/2,(z0+z1)/2
        dx,dy,dz=x1-x0,y1-y0,z1-z0
        small = dx<cs*3 and dz<cs*3 and abs(cz+depth)<cs
        onbox = (abs(x0+box)<eps or abs(x1-box)<eps or abs(y0+box)<eps or
                 abs(y1-box)<eps or abs(z0+box)<eps or abs(z1-box)<eps)
        if small and abs(cy+cl)<eps: tags["gamma0"].append(t)
        elif small and abs(cy-cl)<eps: tags["gamma1"].append(t)
        elif abs(cx+cs/2)<eps and dy>1: tags["alpha0"].append(t)   # -x face of coil
        elif abs(cx-cs/2)<eps and dy>1: tags["alpha1"].append(t)   # +x face
        elif abs(cz-(-depth-cs/2))<eps and dy>1: tags["beta0"].append(t)  # -z
        elif abs(cz-(-depth+cs/2))<eps and dy>1: tags["beta1"].append(t)  # +z
        elif onbox: tags["outer"].append(t)
    # size fields
    def bf(vin,xa,xb,ya,yb,za,zb,th=0.3):
        f=gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(f,"VIn",vin); gmsh.model.mesh.field.setNumber(f,"VOut",lc_far)
        for k,v in (("XMin",xa),("XMax",xb),("YMin",ya),("YMax",yb),("ZMin",za),("ZMax",zb),("Thickness",th)):
            gmsh.model.mesh.field.setNumber(f,k,v)
        return f
    fs=[bf(lc_coil,-cs,cs,-cl-0.2,cl+0.2,-depth-cs,-depth+cs),
        bf(lc_sample,-1.0,sample_x+0.5,-1.0,1.0,0.5,1.5,0.4)]
    mn=gmsh.model.mesh.field.add("Min"); gmsh.model.mesh.field.setNumbers(mn,"FieldsList",fs)
    gmsh.model.mesh.field.setAsBackgroundMesh(mn)
    for o in ("MeshSizeExtendFromBoundary","MeshSizeFromPoints","MeshSizeFromCurvature"):
        gmsh.option.setNumber("Mesh."+o,0)
    gmsh.option.setNumber("Mesh.MeshSizeMax",lc_far)
    ids={}
    gmsh.model.addPhysicalGroup(3,[air],1); gmsh.model.setPhysicalName(3,1,"air"); ids["air"]=1
    gmsh.model.addPhysicalGroup(3,[coil],2); gmsh.model.setPhysicalName(3,2,"coil"); ids["coil"]=2
    bid=1
    for name in ("outer","gamma0","gamma1","alpha0","alpha1","beta0","beta1"):
        assert tags[name], f"no faces for {name}"
        gmsh.model.addPhysicalGroup(2,tags[name],bid); gmsh.model.setPhysicalName(2,bid,name)
        ids[name]=bid; bid+=1
    gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
    gmsh.model.mesh.generate(3)
    ne=sum(len(e) for e in gmsh.model.mesh.getElements(3)[1])
    gmsh.write(msh); gmsh.finalize()
    print("ids:", ids, "elements:", ne)
    return ids

if __name__=="__main__":
    build(sys.argv[1])
