# YADS: Yet Another Darcy Solver
## Context 
This library has been developed during Antoine Lechevallier's thesis. It is a sandbox with two main functionalities: 

    - run reservoir simulations for CO2 storage
    - couple simulations with machine learning models

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

## References:
Antoine Lechevallier PhD report:
Antoine Lechevallier. Physics Informed Deep Learning : Applications to well opening and closing events. Nonlinear Sciences [physics]. Sorbonne Université, 2024. English. ⟨NNT : 2024SORUS062⟩. ⟨tel-04607497⟩
Articles: 



