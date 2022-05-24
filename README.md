# BOPlanner
 This repository contains the company code with the preprint: "High-dimensional Automated Radiation Therapy Treatment Planning via Bayesian Optimization". The code serves to perform automated hyperparameter tuning for IMRT treatment planning. The implemented automated framework was integrated with Eclipse TPS for FMO, leaf sequencing and dose calculation. 

Tested with the Eclipse TPS 15.6 and 16.1. 

## Strcuture:
- [/bo_planner/](/bo_planner/): contains source files realizing key functionalities.
- [main.py](main.py) is the running script to perform the hyperparameter tuning.
- [/results/](results) contains data and script reproducing figures in the manuscript.
- [/config/](/config/) configuration files defining the treatment planning [environment](/config/env_config.json), [clinical goals](/config/PQM/), and [planning parameters](/config/prescriptions/)(i.e. search space). [Objective constraints](/config/prescriptions/Rectum/rectum_constraint.txt) can be defined to limit the search space.

## Requirements 
You need Python 3.7.3 or later to run BOPlanner.

The required Python dependencies are in [requirements.txt](requirements.txt).

## Getting Started
### Define plan quality metrics (PQM)
 Clinical goals are saved in a CSV file as PQM with Structure, Type, Volume, Doselimit, LimitType, Priority, etc. The types of PQM include, min/max dose, mean dose, dose-volume parameters, and dose spillage (R<sub>50%</sub>, R<sub>90%</sub>). PQM grouped the clinical goals into several tiers indexed by Priority where lower tiers, such as 1, have greater importance. The format of PQM can refer to [PQM](/config/PQM/Rectum/PQM.csv).
 
### Set up search space and constraints
- Search space: planning parameters and adjustment ranges are stored in a CSV file. The adjustable planning parameters included both dose objectives and their corresponding weights, using closed intervals to indicate the range of adjustment. And fixed plan parameters were defined by fixed floats. The format of search space can refer to [search space](/config/prescriptions/Rectum/34D.csv).

- Constraints: in order for the planning parameters to be valid and meaningful, constraints need to be added, which only exist for DoseLimit. The format of constraints can refer to [objective constraints](/config/prescriptions/Rectum/rectum_constraint.txt).

### Run BOPlanner
 [tutorial_for_BOPlanner.ipynb](tutorial_for_BOPlanner.ipynb) shows how to use BOPlanner and results of BOPlanner.

## Paper
More details of this work are available in our preprint ["High-dimensional Automated Radiation Therapy Treatment Planning via Bayesian Optimization"](https://arxiv.org/abs/2205.10980) .

## Acknowledgement:

This work is greatly inspired by the [Raybay](https://github.com/kels271828/RayBay/) project. 
