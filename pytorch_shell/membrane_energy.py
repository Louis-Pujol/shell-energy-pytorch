import torch


def membrane_energy(
    points_undef: torch.Tensor,
    points_def: torch.Tensor,
    triangles: torch.Tensor,
):
    """Compute the membrane energy of the mesh.

    The mathematical formulation is given by equation (8) of:
    https://ddg.math.uni-goettingen.de/pub/HeRuWaWi12_final.pdf

    The implementation provided here is a pytorch version of the cpp
    implementation available at:
    https://gitlab.com/numod/shell-energy/-/blob/main/src/membrane_energy.cpp

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
 
    """
    if (
        points_undef.device != points_def.device
        or points_undef.device != triangles.device
    ):
        raise ValueError("All inputs must be on the same device")

    lambd = 1
    mu = 1

    a = points_def[triangles[:, 0]]  # i
    b = points_def[triangles[:, 1]]  # j
    c = points_def[triangles[:, 2]]  # k

    ei = c - b
    ej = a - c
    ek = a - b

    ei_norm = (ei**2).sum(dim=-1)
    ej_norm = (ej**2).sum(dim=-1)
    ek_norm = (ek**2).sum(dim=-1)

    areas_def = (torch.cross(ei, ej, dim=-1) ** 2).sum(dim=-1) / 4

    a = points_undef[triangles[:, 0], :]
    b = points_undef[triangles[:, 1], :]
    c = points_undef[triangles[:, 2], :]

    ei = c - b
    ej = a - c
    ek = a - b

    areas_undef = (torch.cross(ei, ej, dim=-1) ** 2).sum(dim=-1) / 4

    trace = (
        ei_norm * (ej * ek).sum(dim=-1)
        + ej_norm * (ek * ei).sum(dim=-1)
        - ek_norm * (ei * ej).sum(dim=-1)
    )

    return torch.sum(
        (mu * trace / 8 + lambd * areas_def / 4) / areas_undef.sqrt()
        - areas_undef.sqrt()
        * ((mu / 2 + lambd / 4) * torch.log(areas_def / areas_undef) + mu + lambd / 4),
        dim=0,
    )
