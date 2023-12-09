import torch
import numpy as np
import igl

def membrane_energy(v_undef, v_def, f):

    lambd = 1
    mu = 1

    points_undef = torch.from_numpy(v_undef).float()
    points_def = torch.from_numpy(v_def).float()
    triangles = torch.from_numpy(f).int()

    a = points_def[triangles[:, 0], :] # i
    b = points_def[triangles[:, 1], :] # j 
    c = points_def[triangles[:, 2], :] # k

    ei = c - b
    ej = a - c
    ek = a - b

    ei_norm = (ei ** 2).sum(dim=1)
    ej_norm = (ej ** 2).sum(dim=1)
    ek_norm = (ek ** 2).sum(dim=1)

    areas_def = (torch.cross(ei, ej, dim=1) ** 2).sum(dim=1) / 4
    # replace zeros by inf to avoid division by zero
    # areas_undef[areas_undef.sqrt() < 1e15] = float("inf")

    a = points_undef[triangles[:, 0], :]
    b = points_undef[triangles[:, 1], :]
    c = points_undef[triangles[:, 2], :]

    ei = c - b
    ej = a - c
    ek = a - b

    areas_undef = (torch.cross(ei, ej, dim=1) ** 2).sum(dim=1) / 4
    # replace zeros by inf to avoid division by zero
    # areas_def[areas_def.sqrt() < 1e15] = float("inf")

    trace = (
        ei_norm * (ej * ek).sum(dim=1)
        + ej_norm * (ek * ei).sum(dim=1)
        - ek_norm * (ei * ej).sum(dim=1)
    ) 

    return torch.sum(
        (mu * trace / 8  + lambd * areas_def / 4) / areas_undef.sqrt()
        - areas_undef.sqrt() * (
            (mu / 2 + lambd / 4) * torch.log(areas_def / areas_undef)
            + mu + lambd / 4
            )
    )


def bending_energy(v_undef, v_def, f, E, EMAP, EF, EI):
    """Compute the bending energy of the mesh.

    Page 4 in :
    https://ddg.math.uni-goettingen.de/pub/HeRuSc14.pdf

    cpp implementation:
    https://gitlab.com/numod/shell-energy/-/blob/main/src/bending_energy.cpp

    """
    points_undef = torch.from_numpy(v_undef).float()
    points_def = torch.from_numpy(v_def).float()
    triangles = torch.from_numpy(f).int()
    E = torch.from_numpy(E)
    EMAP = torch.from_numpy(EMAP)
    EF = torch.from_numpy(EF)
    EI = torch.from_numpy(EI)

    import time
    import igl

    E, EMAP, EF, EI = igl.edge_flaps(triangles.numpy())
    E = torch.from_numpy(E)
    EMAP = torch.from_numpy(EMAP)
    EF = torch.from_numpy(EF)
    EI = torch.from_numpy(EI)
    
    not_boundary = (EF[:, 0] != -1) * (EF[:, 1] != -1)
    E = E[not_boundary]
    EF = EF[not_boundary]
    EI = EI[not_boundary]
    
    pi, pj = E[:, 0], E[:, 1]
    pk = triangles[EF[:, 0], EI[:, 0]]
    pl = triangles[EF[:, 1], EI[:, 1]]

    # Deformed geometry
    Pi = points_def[pi, :]
    Pj = points_def[pj, :]
    Pk = points_def[pk, :]
    Pl = points_def[pl, :]
    delTheta = _dihedral_angle(Pi, Pj, Pk, Pl)

    # Undeformed geometry
    Pi = points_undef[pi, :]
    Pj = points_undef[pj, :]
    Pk = points_undef[pk, :]
    Pl = points_undef[pl, :]
    delTheta = delTheta - _dihedral_angle(Pi, Pj, Pk, Pl)

    vol = (
        (torch.cross(Pk - Pj, Pi - Pk)).norm(dim=1) / 2
        + (torch.cross(Pl - Pi, Pj - Pl)).norm(dim=1) / 2
    )

    elengthSqr = ((Pj - Pi) ** 2).sum(dim=1)

    divide = (elengthSqr / vol)
    delTheta_square = delTheta ** 2
    tmp = 3 * delTheta_square * divide

    sum_tmp = tmp.sum()
    return sum_tmp

def _dihedral_angle(Pi, Pj, Pk, Pl):

    nk = torch.cross(Pk - Pj, Pi - Pk)
    nk = nk / nk.norm(dim=1).unsqueeze(1)
    nl = torch.cross(Pl - Pi, Pj - Pl)
    nl = nl / nl.norm(dim=1).unsqueeze(1)

    cross_prod = torch.cross(nk, nl)
    aux = torch.clip(
        (nk * nl).sum(dim=1),
        min=-1,
        max=1,
    )
    
    tmp = (cross_prod * (Pj - Pi)).sum(dim=1)
    return (torch.sign(tmp)) * torch.acos(aux)