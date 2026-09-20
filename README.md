# Adaptive-Finite-Difference-Method---Diffusion-Equation
An adaptive finite difference method solver of the diffusion equation in 2D using DUNE. It is important to note that this project is merely an exploration on the use of the DUNE software. The author does not claim that this is an efficient way to solve the diffusion equation. There is still some advantages of starting with a simple FDM method and expanding it to an adaptive solver.

To be able to use the code please first install DUNE:
<a href=https://gitlab.maths.lu.se/dune/installdune#installdune> How to install DUNE </a>

A detailed explanation of the theory can be read in the author's thesis paper: Adaptive Finite Difference Methods and Implementation in the Software DUNE
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22841990.svg)](https://doi.org/10.5281/zenodo.22841990).

The python script for the solver is found in the file "PoissonSolverwT.py" and an implementation is in the notebook "AdaptiveFDM.ipynb". NOTE! To be able to run the code, you need to first activate the DUNE environment.

```bash
source ./dune-env/bin/activate
```
