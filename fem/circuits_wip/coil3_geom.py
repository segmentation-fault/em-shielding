"""Three straight stranded coils (cables) in an air box, each with 6 tagged faces
for the direction chain. Bodies: air=1, coil0/1/2 = 2/3/4.
Boundaries: outer=1, then per cable k: gamma0,gamma1,alpha0,alpha1,beta0,beta1.
Returns the id dict.
"""
import sys, gmsh, numpy as np

def build(msh, xcs=(-0.25,0.0,0.25), box=5.0, cl=3.5, cs=0.2, depth=1.5,
          lc_far=1.2, lc_coil=0.07, lc_sample=0.13, sample_x=4.2):
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("coil3"); occ = gmsh.model.occ
    eps=1e-3
    b=occ.addBox(-box,-box,-box,2*box,2*box,2*box)
    cbars=[occ.addBox(xc-cs/2,-cl,-depth-cs/2,cs,2*cl,cs) for xc in xcs]
    occ.fragment([(3,b)],[(3,c) for c in cbars]); occ.synchronize()
    coils={}; air=None
    for (_,t) in gmsh.model.getEntities(3):
        x0,y0,z0,x1,y1,z1=gmsh.model.getBoundingBox(3,t)
        cx=(x0+x1)/2
        if (x1-x0)<1 and abs((z0+z1)/2+depth)<0.1:
            k=int(np.argmin([abs(cx-x) for x in xcs])); coils[k]=t
        else: air=t
    tags={"outer":[]}
    for k in range(len(xcs)):
        for f in ("gamma0","gamma1","alpha0","alpha1","beta0","beta1"):
            tags[f"{f}_{k}"]=[]
    for (_,t) in gmsh.model.getEntities(2):
        x0,y0,z0,x1,y1,z1=gmsh.model.getBoundingBox(2,t)
        cx,cy,cz=(x0+x1)/2,(y0+y1)/2,(z0+z1)/2
        dx,dy,dz=x1-x0,y1-y0,z1-z0
        near=abs(cz+depth)<cs and dx<cs*3 and dz<cs*3
        onbox=(abs(x0+box)<eps or abs(x1-box)<eps or abs(y0+box)<eps or
               abs(y1-box)<eps or abs(z0+box)<eps or abs(z1-box)<eps)
        k=int(np.argmin([abs(cx-x) for x in xcs]))
        if near and abs(cy+cl)<eps: tags[f"gamma0_{k}"].append(t)
        elif near and abs(cy-cl)<eps: tags[f"gamma1_{k}"].append(t)
        elif dy>1 and abs(cz+depth)<cs and abs(cx-(xcs[k]-cs/2))<eps: tags[f"alpha0_{k}"].append(t)
        elif dy>1 and abs(cz+depth)<cs and abs(cx-(xcs[k]+cs/2))<eps: tags[f"alpha1_{k}"].append(t)
        elif dy>1 and abs(cz-(-depth-cs/2))<eps: tags[f"beta0_{k}"].append(t)
        elif dy>1 and abs(cz-(-depth+cs/2))<eps: tags[f"beta1_{k}"].append(t)
        elif onbox: tags["outer"].append(t)
    def bf(vin,xa,xb,ya,yb,za,zb,th=0.3):
        f=gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(f,"VIn",vin); gmsh.model.mesh.field.setNumber(f,"VOut",lc_far)
        for k2,v in (("XMin",xa),("XMax",xb),("YMin",ya),("YMax",yb),("ZMin",za),("ZMax",zb),("Thickness",th)):
            gmsh.model.mesh.field.setNumber(f,k2,v)
        return f
    sp=max(abs(x) for x in xcs)
    fs=[bf(lc_coil,-sp-cs,sp+cs,-cl-0.2,cl+0.2,-depth-cs,-depth+cs),
        bf(lc_sample,-1.0,sample_x+0.5,-1.0,1.0,0.5,1.5,0.4)]
    mn=gmsh.model.mesh.field.add("Min"); gmsh.model.mesh.field.setNumbers(mn,"FieldsList",fs)
    gmsh.model.mesh.field.setAsBackgroundMesh(mn)
    for o in ("MeshSizeExtendFromBoundary","MeshSizeFromPoints","MeshSizeFromCurvature"):
        gmsh.option.setNumber("Mesh."+o,0)
    gmsh.option.setNumber("Mesh.MeshSizeMax",lc_far)
    ids={}
    gmsh.model.addPhysicalGroup(3,[air],1); gmsh.model.setPhysicalName(3,1,"air"); ids["air"]=1
    for k in range(len(xcs)):
        gmsh.model.addPhysicalGroup(3,[coils[k]],2+k); gmsh.model.setPhysicalName(3,2+k,f"coil{k}"); ids[f"coil{k}"]=2+k
    bid=1
    gmsh.model.addPhysicalGroup(2,tags["outer"],bid); gmsh.model.setPhysicalName(2,bid,"outer"); ids["outer"]=bid; bid+=1
    for k in range(len(xcs)):
        for f in ("gamma0","gamma1","alpha0","alpha1","beta0","beta1"):
            key=f"{f}_{k}"; assert tags[key], f"no faces {key}"
            gmsh.model.addPhysicalGroup(2,tags[key],bid); gmsh.model.setPhysicalName(2,bid,key); ids[key]=bid; bid+=1
    gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
    gmsh.model.mesh.generate(3)
    ne=sum(len(e) for e in gmsh.model.mesh.getElements(3)[1])
    gmsh.write(msh); gmsh.finalize()
    print("elements:", ne, "| coils:", [ids[f"coil{k}"] for k in range(len(xcs))])
    return ids
