from .bending_energy import bending_energy
from .membrane_energy import membrane_energy


def shell_energy(
    points_undef,
    points_def,
    triangles,
    weight=0.001,
):
    """Compute the shell energy.

    The shell energy is defined as the sum of the membrane and weight * bending
    energies.

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

    return membrane_energy(
        points_undef=points_undef,
        points_def=points_def,
        triangles=triangles,
    ) + weight * bending_energy(
        points_undef=points_undef,
        points_def=points_def,
        triangles=triangles,
    )
