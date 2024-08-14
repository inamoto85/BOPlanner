import atexit
import os
import numpy as np
import torch
import pandas as pd
import pyesapi as api
import json
from nm_planner.metric_NM import PlanQualityMetric
from nm_planner.opt_backend_NM import OptSimulator, OptRunner
import nm_planner.opt_space_NM as opt_space
from scipy.optimize import minimize
from nm_planner.initial_NM import init_data


def main(patient_id=None,
         course_id="C1", plan_id="sobol",
         goal_fn="./config/PQM/Rectum/PQM.csv",
         obj_fn="./config/prescriptions/Rectum/34D.csv",
         fn_fixed="./config/prescriptions/Rectum/34D_template.csv",
         n_init=35,
         maxiter=40,
         maxfev=40,
         opt_dir=None,
         dvh_dir=None,
         adjust_important_pars=False,
         **kwargs
         ):
    """Main routine for the automated planning.
    The total number of queried samples are `n_init + n_batch * batch_size`
    Args:
        n_batch: total number of trial batches
        patient_id: patient ID
        course_id: course ID
        plan_id: Plan ID
        model_select: BO model to use
        goal_fn: filename defining the clinical goals for PQM
        obj_fn: filename defining the objective space for TPS
        const_fn: filename defining the constraints
        fn_fixed: filename defining values of planning parameters
        n_init: initial trial numbers, currently using Sobol sequences
        batch_size: size of each trial batch
        opt_dir: file path for saving results of parameters and lengthscales
        dvh_dir: file path for saving dvhs
        adjust_important_pars: select the method of planning parameters adjustment
    Return:
        None
    """
    config_fn = kwargs.pop("env_fn", "./config/env_config.json")
    with open(config_fn, "r") as buf:
         config = json.load(buf)
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    goal_df = pd.read_csv(goal_fn)
    opt_prefix = "./opt_res/{}".format(patient_id) + opt_dir
    dvh_prefix = "./opt_res/{}".format(patient_id) + dvh_dir
    if not os.path.exists(opt_prefix):
        os.makedirs(opt_prefix)
    if not os.path.exists(dvh_prefix):
        os.makedirs(dvh_prefix)
    if adjust_important_pars == True:
        lenscale_fn = "./opt_res/{}/util/hyperpars.csv".format(patient_id)
        important_pars = opt_space.important_pars(lenscale_fn, 5)
        p_dict, f_dict = opt_space.parse_obj_lenscal(fn_fixed, obj_fn, important_pars)
    else:
        f_df, p_dict, f_dict = opt_space.parse_obj(obj_fn)
    struct_roi = {i.split("_")[0] for i in goal_df["Structure"]}
    # add BODY for further extractions of total dvh
    struct_roi.add("BODY")
    app = api.CustomScriptExecutable.CreateApplication("BayOpt")
    atexit.register(app.Dispose)
    # initialize simulator
    patient = app.OpenPatientById(patient_id)
    patient.BeginModifications()
    plan = patient.CoursesLot(course_id).PlanSetupsLot(plan_id)
    plan.SetCalculationModel(api.CalculationType.PhotonVolumeDose, config["DCA"])
    plan.SetCalculationModel(api.CalculationType.PhotonIMRTOptimization, config["OA"])
    plan.SetCalculationModel(api.CalculationType.PhotonVMATOptimization, config["OA"])
    plan.SetCalculationModel(api.CalculationType.PhotonLeafMotions, config["LMC"])
    plan.SetCalculationOption(config["DCA"], "UseGPU", "Yes")
    plan.SetCalculationOption(config["OA"], "UseGPU", "Yes")

    # initial for N-M
    init_x, init_utility = init_data(patient_id, n_init)

    # PQM
    plan_quality = PlanQualityMetric(patient_id, plan, struct_roi, goal_df, p_dict, f_dict, opt_prefix, dvh_prefix)

    # N-M
    res = minimize(plan_quality.function_PQM, np.array(init_x[0]), method='Nelder-Mead', bounds=list(p_dict.values()), tol=1e-6,
                   options={'initial_simplex': np.array(init_x), 'maxiter': maxiter, 'maxfev': maxfev})
    print("res.x: {}".format(res.x))
    print("res.fun: {}".format(res.fun))

    # cal & reopt
    keys = list(p_dict.keys())
    dict_pars = {}
    k = 0
    for i in keys:
        dict_pars[i] = res.x[k]
        k += 1
    plan_opter = OptSimulator(plan=plan, patient=patient_id, roi=struct_roi, p_dict=p_dict)
    cal_runner = OptRunner(simulator=plan_opter, calc_plan=True)
    plan_opter.set_pars(dict_pars)
    cal_dvh = cal_runner.run()
    utility_cal = plan_quality._evaluate(cal_dvh)
    plan_quality.save_data.save_opt_data(list(res.x), utility_cal)
    reopt_runner = OptRunner(simulator=plan_opter, calc_plan=True, reopt=True, normalize=True)
    reopt_dvh = reopt_runner.run()
    utility_reopt = plan_quality._evaluate(reopt_dvh)
    plan_quality.save_data.save_opt_data(list(res.x), utility_reopt)

    app.SaveModifications()
    app.ClosePatient()
    app.Dispose()
    print("optimization completed")


if __name__ == "__main__":
    with open("./config/pat_names.txt", "r") as buf:
         pat_list = list(buf.readline().strip().split(" "))
    for pa in pat_list:
        print(pa)
        main(patient_id=pa,
             plan_id="test",
             opt_dir="/util_nm/",
             dvh_dir="/dvh_nm/",
             maxiter=122,
             maxfev=122,
             adjust_important_pars=False)
        print(pa + " NM is ok !")




