#%%
import torch

def sixDOF(vec):
    R0 = vec[...,:3]/(vec[...,:3]**2).sum(-1,keepdim=True).sqrt()

    a2 = vec[...,3:] - (R0*vec[...,3:]).sum(-1,keepdim=True)*R0
    R1 = a2/(a2**2).sum(-1,keepdim=True).sqrt()

    R2 = torch.cross(R0,R1,dim=-1)
    
    R = torch.stack([R0,R1,R2],dim=-2)
    R = torch.eye(3,3).unsqueeze(0).unsqueeze(0).repeat(R.shape[0],R.shape[1],1,1).to(vec.device)
    return R


def coordRay(spatialVec,rays):
    rotation = sixDOF(spatialVec[...,:6]) #(b,o,3,3)
    if len(rays.shape) == 2:
        rays = rays.unsqueeze(0).repeat(spatialVec.shape[0],1,1)
    coord_direction = torch.einsum('borc,bRc->bRor',rotation,rays)#(b,r,o,3)

    position = spatialVec[...,6:].unsqueeze(1).repeat(1,rays.shape[1],1,1)#(b,r,o,3)

    rays = rays.unsqueeze(-2).repeat(1,1,position.shape[2],1)#(b,r,o,3)
    
    cross_rays_position = torch.cross(position,rays,dim=-1)#(b,r,o,3)
    
    coord_spatial = torch.einsum('borc,bRoc->bRor',rotation,cross_rays_position)#(b,r,o,3)

    return torch.concat([coord_direction,coord_spatial],dim=-1)#(b,r,o,3)

# %%

if __name__=="__main__":
    a = torch.randn(128*128,6)
    R = sixDOF(a)
    print( a)
    print(R)
    print((R[:,0]*R[:,1]).sum(-1))
    print((R[:,1]*R[:,2]).sum(-1))
    print((R[:,2]*R[:,1]).sum(-1))


    # %%
    rays = torch.randn(128*128,3)
    spatialVec = torch.randn(64,20,9)
    coordRay(spatialVec,rays)


# %%
