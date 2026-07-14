# JANC: A cost-effective, differentiable compressible reacting flow solver featured with JAX-based adaptive mesh refinement

JANC, as the abbreviation for “Joint Analyzer for Numerical Combustion”, is a fully-differentiable compressible reacting flow solver based on [JAX-AMR](https://github.com/AlliSim/JAX-AMR).

Authors:
- [Haocheng Wen](https://github.com/thuwen)
- [Faxuan Luo](https://github.com/luofx23)

> ⚠️**NOTE**: This repository has been transferred from the organization `JA4S` to `AllinSim`.
> Old GitHub address will keep redirecting temporarily; please update your remote‑url to `git@github.com:AllinSim/JANC.git`.

## Basic features of JANC
- Conjunction with JAX-AMR, allowing cost-effective large-scale simulations.
- Adoption of structured Cartesian grid, dimensionless equations,  high-order finite difference method, point-implicit chemical source advancing in the solver.
- Inheriting the basic features of JAX, including fully differentiable, compatible with CPUs/GPUs/TPUs computation, and convenient parallel management.
- Programmed by Python, allowing rapid and efficient prototyping of projects.

## Physical models and numerical methods
- Adaptive mesh refinenment (JAX-AMR)
- Dimensionless computation
- Explicit time advancing (RK3)
- High-order spatial reconstruction (WENO-5)
- Riemann solvers (Lax-Friedrichs)
- Point-implicit chemical source advancing
- CPU/GPU/TPU capability
- Parallel computation on GPU/TPU (only for the core solver in current version)

For the details, please refer to our [paper](https://doi.org/10.48550/arXiv.2504.13750).

## Quick Installation
JANC CFD solver and JAX-AMR amr capabilities can be easily installed using pip install git:
```
pip install git+https://github.com/AlliSim/JANC.git
```

## Example

Rotating detonation combustor (RDC) simulation on 1,600,000 grids with 9sp-19r-H2-Air detailed reaction achieved within 45 minutes on single A100 GPU.

Open [janc_basic_example1.ipynb](https://github.com/AlliSim/JANC/blob/main/examples/janc_basic_example1.ipynb) in Google Colab to run the example.

![image](https://github.com/AllinSim/JANC/blob/main/docs/images/RDC_example.gif)

Detonation tube simulation on 4,000,000 grids with 9sp-19r-H2-Air detailed reaction achieved within 1 hour on single A100 GPU.

Open [janc_basic_example2.ipynb](https://github.com/AlliSim/JANC/blob/main/examples/janc_basic_example2.ipynb) in Google Colab to run the example.

![image](https://github.com/AllinSim/JANC/blob/main/docs/images/detonation_tube_example.gif)

## State of the Project

- [x] 2D solver for Euler equations  ✅
- [x] conjuction with the CFD solver ✅
- [x] Parallel compuation for the core solver ✅
- [ ] 3D solver for Navier-Stocks equations (soon)
- [ ] Implicit time advancing (soon)
- [ ] Turbulence model
- [ ] DPM model based on Euler-lagrange method
- [ ] Mixing-precision computation
- [ ] Parallel computation with JAX-AMR

## Citation
JANC: A cost-effective, differentiable compressible reacting flow solver featured with JAX-based adaptive mesh refinement
```
@article{Wen2025,
   author = {Haocheng Wen and Faxuan Luo and Sheng Xu and Bing Wang},
   doi = {10.48550/arXiv.2504.13750},
   journal = {arXiv preprint},
   title = {JANC: A cost-effective, differentiable compressible reacting flow solver featured with JAX-based adaptive mesh refinement},
   year = {2025}
}
```


## License
This project is licensed under the MIT License - see 
the [LICENSE](LICENSE) file or for details https://en.wikipedia.org/wiki/MIT_License.
