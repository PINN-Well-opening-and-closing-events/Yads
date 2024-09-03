![CI Pipeline](https://github.com/PINN-Well-opening-and-closing-events/Yads/actions/workflows/build-test.yml/badge.svg)
![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.11%20%7C%203.12-blue)


# YADS: Yet Another Darcy Solver
## Context 
This library has been developed during Antoine Lechevallier's thesis. It is a sandbox with two main functionalities: 

    - run reservoir simulations for CO2 storage
    - couple simulations with machine learning models


## Tutorials 
Tutorials explaining the main functionalities of the library are available in example/tutorial. The final objective being the modifification of standard numerical schemes and combine it with machine-learning methods. 

## Black-box machine learning model for 1D piston flow

This work is part of the ![CEMRACS 2023](http://smai.emath.fr/cemracs/cemracs23/) on Scientific Machine-Learning. A 1D piston flow is setup with a gradient of pressure from the left to the right. Given different left boundary conditions in pressure and a constant step of time, we generate saturation and pressure profiles at different times. Finally, we learn the solution in pressure and saturation to evolve from a time t to the next timestep t+dt using a neural network. 

Example saturation and pressure profiles obtained at different times with a left boundary pressure of NN MPA. 

![](https://github.com/PINN-Well-opening-and-closing-events/Yads.git/yads/thesis_approaches/CEMRACS/models/article_ressources/figs/sample_piston_darcy.pdf) 

<img src="https://github.com/PINN-Well-opening-and-closing-events/Yads.git/yads/thesis_approaches/CEMRACS/models/article_ressources/figs/sample_piston_darcy.pdf" alt="Logo" width="200"/>


Predicted saturation and pressure profiles at different times given previous timestep solution. 

![](https://github.com/PINN-Well-opening-and-closing-events/Yads.git/yads/thesis_approaches/CEMRACS/models/article_ressources/figs/sample_piston_darcy.pdf) 

Predicted saturation and pressure profiles at different times given initial solution. 

![](https://github.com/PINN-Well-opening-and-closing-events/Yads.git/yads/thesis_approaches/CEMRACS/models/article_ressources/figs/sample_piston_darcy.pdf) 

## Global Hybrid Newton

## Local Hybrid Newton

## Installation (Not working yet)

Still working on proper installation, but numpy alone allows to run most of the scripts 
    
    git clone https://github.com/PINN-Well-opening-and-closing-events/Yads.git
    cd Yads
    conda create --name Yads python=3.12.2
    conda activate Yads
    conda env update --file environment.yml

## Launch tests

    pytest -vvv tests
    coverage run -m pytest -vvv tests
    coverage html 

## Funds
The thesis was financed by:
    - DIM Math Innov (see https://www.dim-mathinnov.fr )
    - IFP Energies Nouvelles (see https://www.ifpenergiesnouvelles.fr )
    - Laboratoire Jacques-Louis Lions (see https://www.ljll.math.upmc.fr )

## References and publications:
Antoine Lechevallier PhD report:
Antoine Lechevallier. Physics Informed Deep Learning : Applications to well opening and closing events. Nonlinear Sciences [physics]. Sorbonne Université, 2024. English. ⟨NNT : 2024SORUS062⟩. ⟨tel-04607497⟩
Articles:




