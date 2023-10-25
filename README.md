# mixpic-bp-quantum-dynamics

This repository contains the mixed picture, tensor network, lazy belief
propagation based quantum dynamics simulation algorithms for the paper
arXiv:2308.05077.

The methods probably require development versions of:

- [quimb](https://quimb.readthedocs.io)
- [cotengra](https://cotengra.readthedocs.io)
- [xyzpy](https://xyzpy.readthedocs.io)
- [autoray](https://autoray.readthedocs.io)

The notebook [`combo_dynamics.ipynb`](combo_dynamics.ipynb) shows the usage
of the following methods:

- MIX (mixed picture)
- PEPO (fully operator evolution)
- PEPS (fully state evolution)
- Exact (via TN simplification and exact contraction)
- CPU-GPU offloading with dynamic slicing for scaling to large $\chi$
