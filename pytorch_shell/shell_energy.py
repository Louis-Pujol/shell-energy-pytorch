from .bending_energy import bending_energy
from .membrane_energy import membrane_energy


def shell_energy(
    points_undef,
    points_def,
    triangles,
    weight=0.001,
):
    return membrane_energy(
        points_undef=points_undef,
        points_def=points_def,
        triangles=triangles,
    ) + weight * bending_energy(
        points_undef=points_undef,
        points_def=points_def,
        triangles=triangles,
    )
