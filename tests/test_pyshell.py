import igl
import numpy as np
import pyshell

def test_run_pyshell_fingers():

    bending_weight = 0.001

    v_undef, f = igl.read_triangle_mesh("data/finger0.ply")
    v_def, _ = igl.read_triangle_mesh("data/finger1.ply")

    f = f.astype(np.int32)

    E, EMAP, EF, EI = igl.edge_flaps(f)

    membrane_energy = pyshell.membrane_energy(v_undef, v_def, f)
    bending_energy = pyshell.bending_energy(v_undef, v_def, f, E, EMAP, EF, EI)
    shell_energy = pyshell.shell_energy(v_undef, v_def, f, E, EMAP, EF, EI, bending_weight)