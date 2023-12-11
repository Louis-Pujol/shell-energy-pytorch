import matplotlib.pyplot as plt
import igl
import numpy as np
import torch
import pyshell
from pytorch_shell import membrane_energy, bending_energy
from time import time

torch.autograd.set_detect_anomaly(True)

# Prepare the data

def running_times(n_batchs):

    v_undefs = []
    v_defs = []

    for i in range(n_batchs):
        i, j = np.random.randint(0, 10, 2)
        v_undefs.append(igl.read_triangle_mesh(f"../data/cactus/cactus{i}.ply")[0])
        v_defs.append(igl.read_triangle_mesh(f"../data/cactus/cactus{j}.ply")[0])

    f = igl.read_triangle_mesh(f"../data/cactus/cactus0.ply")[1]

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

    ### Pyshell ###
    start = time()

    # Compute the membrane energy
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

    ## Compute the bending energy
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

    # Print running time and outputs
    end = time()
    pyshell_runtime = end - start


    ### Pytorch CPU ###

    # Compute the membrane energy
    start = time()
    batch_points_def.requires_grad = True
    batch_points_undef.requires_grad = True
    membrane_energy_torch = membrane_energy(
        points_undef=batch_points_undef, points_def=batch_points_def, triangles=triangles
    )

    # Compute the gradients of the membrane energy
    membrane_energy_torch.sum().backward()
    undef_grad = batch_points_undef.grad
    def_grad = batch_points_def.grad

    # Ceinitialize the gradients
    batch_points_def.grad.zero_()
    batch_points_undef.grad.zero_()

    # Compute the bending energy
    bending_energy_torch = bending_energy(
        points_undef=batch_points_undef, points_def=batch_points_def, triangles=triangles
    )

    # Compute the gradients of the bending energy
    bending_energy_torch.sum().backward()
    undef_grad = batch_points_undef.grad
    def_grad = batch_points_def.grad

    # Print running time and outputs
    end = time()
    torch_cpu_runtime = end - start


    ### Pytorch GPU ###
    batch_points_undef = batch_points_undef.detach().cuda()
    batch_points_def = batch_points_def.detach().cuda()
    triangles = triangles.detach().cuda()

    # Compute the membrane energy
    torch.cuda.synchronize()
    start = time()
    batch_points_def.requires_grad = True
    batch_points_undef.requires_grad = True
    membrane_energy_torch = membrane_energy(
        points_undef=batch_points_undef, points_def=batch_points_def, triangles=triangles
    )

    # Compute the gradients of the membrane energy
    membrane_energy_torch.sum().backward()
    undef_grad = batch_points_undef.grad
    def_grad = batch_points_def.grad

    # Ceinitialize the gradients
    batch_points_def.grad.zero_()
    batch_points_undef.grad.zero_()

    # Compute the bending energy
    bending_energy_torch = bending_energy(
        points_undef=batch_points_undef, points_def=batch_points_def, triangles=triangles
    )

    # Compute the gradients of the bending energy
    bending_energy_torch.sum().backward()
    undef_grad = batch_points_undef.grad
    def_grad = batch_points_def.grad

    # Print running time and outputs
    torch.cuda.synchronize()
    end = time()
    torch_gpu_runtime = end - start

    return pyshell_runtime, torch_cpu_runtime, torch_gpu_runtime


pyshell_runtimes = []
torch_cpu_runtimes = []
torch_gpu_runtimes = []

#  Wake up the GPU
start = time()
X_undef, f = igl.read_triangle_mesh(f"../data/cactus/cactus0.ply")
X_def = igl.read_triangle_mesh(f"../data/cactus/cactus1.ply")[0]

X_def = torch.from_numpy(X_def).float().cuda()
X_undef = torch.from_numpy(X_undef).float().cuda()
triangles = torch.from_numpy(f).int().cuda()

X_def.requires_grad = True
X_undef.requires_grad = True

membrane_energy(
    points_undef=X_undef,
    points_def=X_def,
    triangles=triangles,
).backward()
end = time()
# a = torch.rand(1000).cuda()
# a.requires_grad = True
# a.sum().cos().backward()
torch.cuda.synchronize()
end = time()
print(f"Warming up GPU: {end - start:.3f}")

x = np.arange(1, 21)

for n_batchs in x:
    pyshell_runtime, torch_cpu_runtime, torch_gpu_runtime = running_times(n_batchs)
    pyshell_runtimes.append(pyshell_runtime)
    torch_cpu_runtimes.append(torch_cpu_runtime)
    torch_gpu_runtimes.append(torch_gpu_runtime)

print(x)

print( pyshell_runtimes[-1] / torch_gpu_runtimes[-1] )

plt.plot(x, pyshell_runtimes, label="pyshell")
plt.plot(x, torch_cpu_runtimes, label="torch_cpu")
plt.plot(x, torch_gpu_runtimes, label="torch_gpu")
plt.legend()
plt.show()

