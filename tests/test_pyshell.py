import igl
import numpy as np
import torch
import pyshell
from pytorch_shell import membrane_energy, bending_energy
from time import time


def test_membrane_energy():
    for i in range(5):
        for j in range(i):
            v_undef, f = igl.read_triangle_mesh(f"data/cactus/cactus{i}.ply".format(i))
            v_def, _ = igl.read_triangle_mesh(f"data/cactus/cactus{j}.ply".format(j))
            f = f.astype(np.int32)

            points_undef = torch.from_numpy(v_undef).float()
            points_def = torch.from_numpy(v_def).float()
            triangles = torch.from_numpy(f).int()

            membrane_energy1 = pyshell.membrane_energy(v_undef, v_def, f)
            membrane_energy2 = membrane_energy(
                points_undef=points_undef,
                points_def=points_def,
                triangles=triangles,
            ).numpy()

            print(membrane_energy1, membrane_energy2)
            assert np.allclose(membrane_energy1, membrane_energy2, rtol=1e-2, atol=1e-6)


def test_bending_energy():
    for i in range(5):
        for j in range(i):
            v_undef, f = igl.read_triangle_mesh(f"data/cactus/cactus{i}.ply".format(i))
            v_def, _ = igl.read_triangle_mesh(f"data/cactus/cactus{j}.ply".format(j))
            f = f.astype(np.int32)
            E, EMAP, EF, EI = igl.edge_flaps(f)

            points_undef = torch.from_numpy(v_undef).float()
            points_def = torch.from_numpy(v_def).float()
            triangles = torch.from_numpy(f).int()

            bending_energy1 = pyshell.bending_energy(v_undef, v_def, f, E, EMAP, EF, EI)
            bending_energy2 = bending_energy(
                points_undef=points_undef,
                points_def=points_def,
                triangles=triangles,
            ).numpy()

            print(bending_energy1, bending_energy2)
            assert np.allclose(bending_energy1, bending_energy2, rtol=1e-2, atol=1e-6)
