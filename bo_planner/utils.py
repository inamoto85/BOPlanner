from ax import Experiment
from ax.core.data import Data
from ax.core.generator_run import GeneratorRun
from ax.core.arm import Arm

"""
Utility functions on re-optimization/calculation
at the end of the BO procedure.
"""

def extract_best_params(data: Data, experiment: Experiment):
    """extract the best trial parameters in the current dataset
    Args:
        data: dataset containing parameter-PQM score pairs
        experiment: Experiment object containing previous trials
    Return:
        best parameters and best arm name
    """
    sorted_df = data.df.sort_values("mean", ascending=False)
    arm_name = sorted_df.loc[sorted_df.index[0]]["arm_name"]
    trial_index = sorted_df.loc[sorted_df.index[0]]["trial_index"]
    best_trial = experiment.get_trials_by_indices([trial_index])[0]
    for arm in best_trial.arms:
        if arm.name == arm_name:
            return arm.parameters, arm.name
    raise NotImplementedError("best arm not found, needs to check the trials")

def calculate_plan(experiment, arm_name, params, data):
    """optimize and calculate with the given parameter set
    Args:
        experiment: optimizing environment.
        arm_name: the arm name associated with the parameters.
        params: plan parameters used for optimization.
        data: dataset to save the parameter-score pairs.
    Return:
        dataset containing calculated score based on the best parameter set
    """
    experiment.runner.calc_plan = True
    print(f"best arm name: {arm_name}")
    trial = experiment.new_trial(generator_run=GeneratorRun(arms=[Arm(parameters=params)])).run()
    trial.complete()
    data = Data.from_multiple_data([data, trial.fetch_data()])
    return data, trial

def reopt_calculated_plan(experiment, best_params, data):
    """re-optimize plan based on previous calculated intermediate dose
    Args:
        experiment: Current planning environment.
        arm_name: the arm name associated with the parameters.
        params: plan parameters used for optimization.
        data: dataset to save the parameter-score pairs.
    Return:
        dataset containing reoptimized-calculated score based on the best parameter set 
    """
    experiment.runner.reopt = True
    experiment.runner.normalize = True
    print("re-optimization with calculated dose")
    trial = experiment.new_trial(generator_run=GeneratorRun(arms=[Arm(parameters=best_params)])).run()
    trial.complete()
    data = Data.from_multiple_data([data, trial.fetch_data()])
    return data, trial
