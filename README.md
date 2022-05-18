# bo_planner
 This repository contains the company code with the preprint: "High-dimensional Automated Radiation Therapy Treatment Planning via Bayesian Optimization". The code serves to perform automated hyperparameter tuning for IMRT treatment planning.The implemented automated framework was integrated with Eclipse TPS for FMO, leaf sequencing and dose calculation. Tested with Eclipse 15.6 and 16.1. 
 
### Requirements
Python 3.7, numpy, ax, pandas, PyESAPI, pytorch

## Strcuture:
- [/bo_planner/](/bo_planner/): contains source files realizing key functionalities.
- [main.py](main.py) is the running script to perform the hyperparameter tuning.
- [/results/](results) contains data and script reproducing figures in the manuscript.
- [/config/](/config/) configuration files defining the treatment planning [environment](/config/env_config.json), [clinical goals](/config/PQM/), and [planning parameters](/config/prescriptions/)(i.e. search space). [Objective constraints](/config/prescriptions/Rectum/rectum_constraint.txt) can be defined.

## Acknowledgement:

This work is greatly inspired by the [Raybay](https://github.com/kels271828/RayBay/) project. 
