# shell-energy-pytorch
PyTorch Implementation of [shell energy](https://gitlab.com/numod/shell-energy)

TODO:

- [x] Add an action to setup a python environment with pyshell
- [x] Write pytorch code for shell energy
- [x] Test: do we obtain the same values on examples ?
- [ ] Test: do we obtain the same gradients with `torch.autograd` ? -> Yes for membrane enery, not for bending energy...
- [ ] Benchmark : which one is faster ? Does GPU provides an acceleration ? -> Seem to be true for batches 
