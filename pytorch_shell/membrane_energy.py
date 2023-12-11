import torch


def membrane_energy(
    points_undef: torch.Tensor,
    points_def: torch.Tensor,
    triangles: torch.Tensor,
):
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
    # replace zeros by inf to avoid division by zero
    # areas_undef[areas_undef.sqrt() < 1e15] = float("inf")

    a = points_undef[triangles[:, 0], :]
    b = points_undef[triangles[:, 1], :]
    c = points_undef[triangles[:, 2], :]

    ei = c - b
    ej = a - c
    ek = a - b

    areas_undef = (torch.cross(ei, ej, dim=-1) ** 2).sum(dim=-1) / 4
    # replace zeros by inf to avoid division by zero
    # areas_def[areas_def.sqrt() < 1e15] = float("inf")

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
