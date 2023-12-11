def test_this():
    import igl
    import numpy as np
    import torch
    import pyshell
    from pytorch_shell import membrane_energy, bending_energy
    from time import time

    torch.autograd.set_detect_anomaly(True)

    # Prepare the data

    v_undefs = [
        igl.read_triangle_mesh(f"data/cactus/cactus0.ply")[0],
        igl.read_triangle_mesh(f"data/cactus/cactus1.ply")[0],
        igl.read_triangle_mesh(f"data/cactus/cactus2.ply")[0],
        igl.read_triangle_mesh(f"data/cactus/cactus4.ply")[0],
    ]

    v_defs = [
        igl.read_triangle_mesh(f"data/cactus/cactus3.ply")[0],
        igl.read_triangle_mesh(f"data/cactus/cactus4.ply")[0],
        igl.read_triangle_mesh(f"data/cactus/cactus5.ply")[0],
        igl.read_triangle_mesh(f"data/cactus/cactus6.ply")[0],
    ]

    f = igl.read_triangle_mesh(f"data/cactus/cactus0.ply")[1]

    # Load data (numpy)
    f = f.astype(np.int32)
    E, EMAP, EF, EI = igl.edge_flaps(f)

    # Convert to torch
    shape = v_undefs[0].shape
    batch_points_undef = torch.zeros(shape[0], len(v_undefs), shape[1])
    batch_points_def = torch.zeros(shape[0], len(v_defs), shape[1])
    for i in range(len(v_undefs)):
        batch_points_undef[:, i, :] = torch.from_numpy(v_undefs[i]).float()
        batch_points_def[:, i, :] = torch.from_numpy(v_defs[i]).float()

    triangles = torch.from_numpy(f).int()

    ## Membrane energy

    # Pyshell
    start = time()
    membrane_energies = np.zeros(len(v_undefs))
    membrane_gradient_undef = np.zeros((shape[0], len(v_undefs), shape[1]))
    membrane_gradient_def = np.zeros((shape[0], len(v_undefs), shape[1]))
    for i in range(len(v_undefs)):
        membrane_energies[i] = pyshell.membrane_energy(v_undefs[i], v_defs[i], f)
        membrane_gradient_undef[:, i, :] = pyshell.membrane_undeformed_gradient(
            v_undefs[i], v_defs[i], f
        )
        membrane_gradient_def[:, i, :] = pyshell.membrane_deformed_gradient(
            v_undefs[i], v_defs[i], f
        )
    end = time()
    print("Pyshell: ", end - start)
    print(membrane_energies)

    # Torch
    start = time()
    batch_points_def.requires_grad = True
    batch_points_undef.requires_grad = True
    membrane_energy_torch = membrane_energy(
        points_undef=batch_points_undef,
        points_def=batch_points_def,
        triangles=triangles,
    )

    membrane_energy_torch.sum().backward()
    undef_grad = batch_points_undef.grad
    def_grad = batch_points_def.grad

    end = time()
    print("Torch: ", end - start)
    print(membrane_energy_torch)

    assert torch.allclose(
        membrane_energy_torch,
        torch.from_numpy(membrane_energies).float(),
        atol=1e-5,
        rtol=1e-2,
    )

    print(undef_grad.shape)
    print(torch.from_numpy(membrane_gradient_undef).shape)

    assert torch.allclose(
        undef_grad,
        torch.from_numpy(membrane_gradient_undef).float(),
        atol=1e-5,
        rtol=1e-2,
    )

    assert torch.allclose(
        def_grad,
        torch.from_numpy(membrane_gradient_def).float(),
        atol=1e-5,
        rtol=1e-2,
    )

    ## Bending energy

    bending_energies = np.zeros(len(v_undefs))
    bending_gradient_undef = np.zeros((shape[0], len(v_undefs), shape[1]))
    bending_gradient_def = np.zeros((shape[0], len(v_undefs), shape[1]))
    for i in range(len(v_undefs)):
        bending_energies[i] = pyshell.bending_energy(
            v_undef=v_undefs[i], v_def=v_defs[i], f=f, E=E, EMAP=EMAP, EF=EF, EI=EI
        )
        bending_gradient_undef[:, i, :] = pyshell.bending_undeformed_gradient(
            v_undef=v_undefs[i], v_def=v_defs[i], f=f, E=E, EMAP=EMAP, EF=EF, EI=EI
        )
        bending_gradient_def[:, i, :] = pyshell.bending_deformed_gradient(
            v_undef=v_undefs[i], v_def=v_defs[i], f=f, E=E, EMAP=EMAP, EF=EF, EI=EI
        )

    print(bending_energies)

    # Torch
    batch_points_def.requires_grad = True
    batch_points_undef.requires_grad = True
    bending_energy_torch = bending_energy(
        points_undef=batch_points_undef,
        points_def=batch_points_def,
        triangles=triangles,
    )
    bending_energy_torch.sum().backward()
    undef_grad = batch_points_undef.grad
    def_grad = batch_points_def.grad

    print(bending_energy_torch)

    # Assert that energies are the same
    assert torch.allclose(
        bending_energy_torch,
        torch.from_numpy(bending_energies).float(),
        atol=1e-5,
        rtol=1e-2,
    )

    # # Assert that gradients are the same (deformated)
    # assert torch.allclose(
    #     def_grad,
    #     torch.from_numpy(bending_gradient_def).float(),
    #     atol=1e-5,
    #     rtol=1e-2,
    # )

    # Assert that gradients are the same (undeformated)
    # assert torch.allclose(
    #     undef_grad,
    #     torch.from_numpy(bending_gradient_undef).float(),
    #     atol=1e-5,
    #     rtol=1e-2,
    # )
