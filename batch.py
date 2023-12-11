import igl
import numpy as np
import torch
import pyshell
from pytorch_shell import membrane_energy, bending_energy
from time import time

torch.autograd.set_detect_anomaly(True)

# Prepare the data

n_batchs = 10

v_undefs = []
v_defs = []

for i in range(n_batchs):
    i, j = np.random.randint(0, 10, 2)
    v_undefs.append(igl.read_triangle_mesh(f"data/cactus/cactus{i}.ply")[0])
    v_defs.append(igl.read_triangle_mesh(f"data/cactus/cactus{j}.ply")[0])

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
gpu = False

if gpu:
    batch_points_undef = batch_points_undef.cuda()
    batch_points_def = batch_points_def.cuda()
    triangles = triangles.cuda()
    # Wake up the GPU
    start = time()
    bending_energy(
    points_def=batch_points_def[:, 0, :], points_undef=batch_points_undef[:, 0, :], triangles=triangles
    )
    end = time()
    print(f"Running time for GPU initialization: {end - start:.3f}")

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
print(f"Running time for pyshell implementation: {end - start:.3f}")
# print(f"Membrane energies: {membrane_energies}")
# print(f"Bending energies: {bending_energies}")


### Pytorch ###


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
print(f"Running time for pytorch implementation: {end - start:.3f}")
# print(f"Membrane energies: {membrane_energy_torch}")
# print(f"Bending energies: {bending_energy_torch}")


exit()

# Assert that bending are the same
assert torch.allclose(
    bending_energy_torch,
    torch.from_numpy(bending_energies).float(),
    atol=1e-5,
    rtol=1e-2,
)

# Assert that membrane are the same
assert torch.allclose(
    membrane_energy_torch,
    torch.from_numpy(membrane_energies).float(),
    atol=1e-5,
    rtol=1e-2,
)

# Assert that membrane energy gradients are the same (def)
assert torch.allclose(
    undef_grad,
    torch.from_numpy(membrane_gradient_undef).float(),
    atol=1e-5,
    rtol=1e-2,
)

# Assert that membrane energy gradients are the same (undef)
assert torch.allclose(
    def_grad,
    torch.from_numpy(membrane_gradient_def).float(),
    atol=1e-5,
    rtol=1e-2,
)


# Assert that bending gradients are the same (def)
assert torch.allclose(
    def_grad,
    torch.from_numpy(bending_gradient_def).float(),
    atol=1e-5,
    rtol=1e-2,
)

# Assert that bending gradients are the same (undef)
assert torch.allclose(
    undef_grad,
    torch.from_numpy(bending_gradient_undef).float(),
    atol=1e-5,
    rtol=1e-2,
)
