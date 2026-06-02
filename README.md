# Finite Element Ion Channel Modeling

![Poisson-Boltzmann experiment matrix](poisson_boltzmann_matrix.png)

## Description

This repository contains the Python code for a series of numerical experiments
regarding partial differential equations and ion-channel modeling:

- **Manufactured-solution Poisson experiment:** verifies a basic FEniCSx
  implementation against a known quadratic solution and measures how its error
  changes as the mesh is refined.
- **Three-dimensional cube verification:** compares a finite-element solution
  of the linearized Poisson-Boltzmann equation with Holst's screened
  single-charge formula after approximating the point charge with a Gaussian.
- **Two-domain Poisson-Boltzmann experiment:** models a low-permittivity
  molecular region inside a solvent region and studies how the computed
  potential changes across their interface and under mesh refinement.

**Stack:** Python | FEniCSx | Gmsh | UFL | MPI | NumPy | Matplotlib | PyVista

For further context, read the two papers in this
[Google Drive folder](https://drive.google.com/drive/folders/1dkOa-wnW3BBUInCp_A_2ZXw6GaKBVa_0?usp=drive_link):

- ***The Poisson Equation and Its Role in Ion Channel Modeling:*** introduces
  the Poisson equation, numerical methods with FEniCSx, and the
  Poisson-Boltzmann and Poisson-Nernst-Planck equations used in ion-channel
  modeling.
- ***A Poisson Boltzmann Numerical Experiment:*** extends the earlier study
  with a numerical implementation of the linearized Poisson-Boltzmann equation,
  including the cube verification and two-domain experiment.

## Motivation

The material in this repository is the result of an independent study
conducted with [Professor Zhen Chao](https://mathematics.wwu.edu/people/chaoz2)
of Western Washington University, whose research includes mathematical models
for ion-channel proteins. Throughout the nearly year-long study, we started
with the basics of partial differential equations and numerical methods,
including finite differences, and continued until we reached the final
deliverable: simulating the Poisson-Boltzmann equation, a PDE often used in
ion-channel modeling, with the finite-element method in Python.

## Quick Start

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate finite-element-ion-channel-modeling
```

Run the introductory Poisson benchmark:

```bash
python poisson_fundamentals.py
```

Run the linearized Poisson-Boltzmann experiments:

```bash
python poisson_boltzmann.py
```

Generated meshes, plots, summaries, and solver output are written to
`results/`.

## Author

Built by Brandon Connely.
