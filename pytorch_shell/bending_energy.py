import torch
import igl
import numpy as np


def bending_energy(
    points_undef: torch.Tensor,
    points_def: torch.Tensor,
    triangles: torch.Tensor,
):
    """Compute the bending energy of the mesh.

    The mathematical formulation is given in page 4 of:
    https://ddg.math.uni-goettingen.de/pub/HeRuSc14.pdf

    The implementation provided here is a pytorch version of the cpp
    implementation available at:
    https://gitlab.com/numod/shell-energy/-/blob/main/src/bending_energy.cpp

    Parameters
    ----------
    points_undef : torch.Tensor
        The undeformed points of the mesh. Shape: (n_points, 3) for a single
        mesh and (n_points, n_meshes, 3) for a batch of meshes in dense
        correspondance.

    points_def : torch.Tensor
        The deformed points of the mesh. Shape: (n_points, 3) for a single
        mesh and (n_points, n_meshes, 3) for a batch of meshes in dense
        correspondance.

    triangles : torch.Tensor
        The triangles of the mesh(es). Shape: (n_triangles, 3).

    weight : float, optional
        The weight of the bending energy. Default: 0.001.

    """
    if (
        points_undef.device != points_def.device
        or points_undef.device != triangles.device
    ):
        raise ValueError("All inputs must be on the same device")

    device = points_undef.device

    E, EMAP, EF, EI = igl.edge_flaps(triangles.cpu().numpy())
    E = torch.from_numpy(E).to(device)
    EMAP = torch.from_numpy(EMAP).to(device)
    EF = torch.from_numpy(EF).to(device)
    EI = torch.from_numpy(EI).to(device)

    not_boundary = (EF[:, 0] != -1) * (EF[:, 1] != -1)
    E = E[not_boundary]
    EF = EF[not_boundary]
    EI = EI[not_boundary]

    pi, pj = E[:, 0], E[:, 1]
    pk = triangles[EF[:, 0], EI[:, 0]]
    pl = triangles[EF[:, 1], EI[:, 1]]

    # Deformed geometry
    Pi = points_def[pi, :]
    Pj = points_def[pj, :]
    Pk = points_def[pk, :]
    Pl = points_def[pl, :]
    delTheta = _dihedral_angle(Pi, Pj, Pk, Pl)

    # Undeformed geometry
    Pi = points_undef[pi, :]
    Pj = points_undef[pj, :]
    Pk = points_undef[pk, :]
    Pl = points_undef[pl, :]
    delTheta = delTheta - _dihedral_angle(Pi, Pj, Pk, Pl)

    vol = (torch.linalg.cross(Pk - Pj, Pi - Pk)).norm(dim=-1) / 2 + (
        torch.linalg.cross(Pl - Pi, Pj - Pl)
    ).norm(dim=-1) / 2

    elengthSqr = ((Pj - Pi) ** 2).sum(dim=-1)

    tmp = 3 * (delTheta**2) * (elengthSqr / vol)

    sum_tmp = tmp.sum(dim=0)
    return sum_tmp


def _dihedral_angle(Pi, Pj, Pk, Pl):
    nk = torch.linalg.cross(Pk - Pj, Pi - Pk, dim=-1)
    nk = nk / nk.norm(dim=-1).unsqueeze(-1)
    nl = torch.linalg.cross(Pl - Pi, Pj - Pl, dim=-1)
    nl = nl / nl.norm(dim=-1).unsqueeze(-1)

    cross_prod = torch.linalg.cross(nk, nl, dim=-1)
    edge_dir = Pj - Pi
    edge_dir = edge_dir / edge_dir.norm(dim=-1).unsqueeze(-1)

    aux = (nk * nl).sum(dim=-1)

    tmp = (cross_prod * edge_dir).sum(dim=-1)

    return torch.atan2(tmp, aux)
