import pandas as pd
import re
from typing import Tuple, Dict, List
import numpy as np
from ax import SearchSpace, RangeParameter, ParameterType, ChoiceParameter
from ax.service.utils.instantiation import constraint_from_str
import pandas as pd
from ax.modelbridge.random import RandomModelBridge
from ax import Experiment
from typing import Dict, cast
from ax.core.data import Data
from ax.core.arm import Arm
from ax.core.types import TTrialEvaluation


def load_constraints(const_fn: str, important_pars=[], f_dict=[]) -> List:
    """helper function to parse constraints
    constraints
    Args:
        const_fn: str, constraint file name
    Return:
        res: list of constraint strings
    """
    def change_name(dict):
        dict_clone = dict.copy()
        keys = list(dict_clone.keys())
        for i in range(len(keys)):
            items = keys[i].split('_')
            if items[-1] == 'Priority':
                new_key = items[0] + '_' + items[1] + '_Priority'
            else:
                new_key = items[0] + '_' + items[1] + '_DoseLimit'
            new_key = new_key.replace('Irrad Volume', 'Irrad_Volume')
            new_key = new_key.replace('Urinary Bladder', 'Urinary_Bladder')
            dict_clone[new_key] = dict_clone.pop(keys[i])
        return dict_clone

    res = []
    res_sample = []
    buf = open(const_fn)
    line = buf.readline().strip()
    if not important_pars:
        while line:
            res.append(line)
            line = buf.readline().strip()
    elif important_pars:
        f_dict_newkeys = change_name(f_dict)
        f_keys = list(f_dict_newkeys.keys())
        while line:
            items = line.split(' <= ')
            if items[0] in f_keys and items[-1] in f_keys:
                line = None
            elif items[0] not in f_keys and items[-1] not in f_keys:
                res.append(line)
            elif items[0] not in f_keys and items[-1] in f_keys:
                line = line.replace(items[-1], str(f_dict_newkeys[items[-1]]))
                res_sample.append(line)
            elif items[0] in f_keys and items[-1] not in f_keys:
                line_new = items[-1] + ' >= ' + str(f_dict_newkeys[items[0]])
                res_sample.append(line_new)
            line = buf.readline().strip()
    return res, res_sample

def init_data(patient_id: str, experiment: Experiment,
              n_init: int, mode="file", file='sobol', generator=None):
    """data initialization for the BO procedure
    Args:
        patient_id: str,
        experiment: Ax experiment object
        n_init: number of initial data to populate
        mode: str, file mode "file" or sobol sequence "sobol" from Ax
        file: str, file directory to look for the initialization file.
        generator: sequence generator
    Return:
        return an experiment object populated with an initial dataset
    """
    if mode == "file":
        # TODO: check path dependencies
        sobolpath = "./opt_res/{}/util_".format(patient_id) + file + "/pars.csv"
        sobol_odf = pd.read_csv(sobolpath)
        metric_name = experiment.optimization_config.objective.metric_names[0]
        for i in range(n_init):
            parameters = sobol_odf.iloc[i, :-1].to_dict()
            value = sobol_odf.iloc[i, -1]
            raw_data = {metric_name: (value, 0)}
            trial = experiment.new_trial().add_arm(Arm(parameters=parameters))
            trial.mark_running(no_runner_required=True)
            evaluations = {trial.arm.name: raw_data}
            data = Data.from_evaluations(evaluations=cast(TTrialEvaluation, evaluations),
                                         trial_index=trial.index)
            experiment.attach_data(data)
            trial.complete()
    elif mode == "sobol":
        assert (isinstance(generator, RandomModelBridge)), "Only support a sobol generator currently."
        for _ in range(n_init):
            trial = experiment.new_trial(generator.gen(1))
            trial.run()
            trial.complete()
    return experiment

def important_pars(fn: str, num=5):
    """select important parameters
    Args:
        fn: str, lengthscale file name
        num: number of important parameters to adjust
    Return:
        return a list of important parameters
    """
    lenscal_df = pd.read_csv(fn)
    lenscals = lenscal_df.iloc[-1].sort_values()[:num]      # TODO: change to [:num]
    pars_index = list(lenscals.index)
    pars = []
    for i in pars_index:
        items = i.split('_')
        stru = items[0]+'_'+items[1]
        if items[-1] == 'Priority':
            stru += '_Priority'
            pars.append(stru)
        else:
            stru += '_DoseLimit'
            pars.append(stru)
    return pars

def parse_obj_lenscal(fn_fixed: str, fn_adjust: str, pars_important: list):
    """parse the optimization csv files
    and construct the optimization search_space for important parameters
    Args:
        fn_fixed: filename containing the template of planning parameters.
        fn_adjust: filename containing the search space specification.
        pars_important: list, importtant parameters
    Return:
        f_dict: Dict, fixed objectives to be set
        p_dict: Dict, modifiable objectives to be set
    """
    fixed_df = pd.read_csv(fn_fixed)
    adjust_df = pd.read_csv(fn_adjust).astype(object)
    par_objs = [i.split('_')[0] + "_" + i.split('_')[1] for i in pars_important]
    p_dict = {}
    f_dict = {}
    for index, row in adjust_df.iterrows():
        # currently only support modifying Doselimit, Volume and priority
        for col in ["DoseLimit", "Volume", "Priority"]:
            if isinstance(row[col], str):
                pars = [float(par) for par in re.findall(r'\d+\.\d+|\d+', row[col])]
                row.loc[col] = pars if len(pars) > 1 else pars[0]
    for index, row in adjust_df.iterrows():
        if row[0] in par_objs:
            adjust_items = [s for s in pars_important if row[0] in s]
            types = [t.split('_')[-1] for t in adjust_items]
            for type in types:
                p_name = form_key(row, type)
                p_dict[p_name] = row[type]
        else:
            continue
    for index, row in fixed_df.iterrows():
        if row[0] not in par_objs:
            for col in ["DoseLimit", "Priority"]:
                p_name = form_key(row, col)
                f_dict[p_name] = row[col]
        elif row[0] in par_objs:
            adjust_items = [s for s in pars_important if row[0] in s]
            types = [t.split('_')[-1] for t in adjust_items]
            for col in ["DoseLimit", "Priority"]:
                if col not in types:
                    p_name = form_key(row, col)
                    f_dict[p_name] = row[col]
    return p_dict, f_dict

def parse_obj(fn: str) -> Tuple[pd.DataFrame, Dict, Dict]:
    """parse the optimization csv files
    and construct the optimization search_space
    Args:
        fn, filename containing the search space specification.
    Return:
        p_df: pd.DataFrame nominal object for the objectives
        f_df: Dict, fixed objectives to be set
        p_dict: Dict, modifiable objectives to be set
    """
    p_df = pd.read_csv(fn).astype(object)
    p_dict = {}
    f_dict = {}
    for index, row in p_df.iterrows():
        # currently only support modifying Doselimit, Volume and priority
        for col in ["DoseLimit", "Volume", "Priority"]:
            if isinstance(row[col], str):
                pars = [float(par) for par in re.findall(r'\d+\.\d+|\d+',
                                                         row[col])]
                row.loc[col] = pars if len(pars) > 1 else pars[0]
    for index, row in p_df.iterrows():
        fixed = True
        for col in ["DoseLimit", "Volume", "Priority"]:
            if isinstance(row[col], list):
                fixed = False
                p_name = form_key(row, col)
                p_dict[p_name] = row[col]
        if fixed is True:
            col = "DoseLimit"
            p_name = form_key(row, col)
            f_dict[p_name] = row[col]
    return p_df, p_dict, f_dict

def form_space(p_dict: Dict, c_strs: List=None, c_sample: List=None, grid=False, grid_size=200):
    """search space definition for BO
    Args:
        p_dict: dict, contains parameter name - value pairs to add in the search space.
        c_strs: search space constraints.
        grid: discretize search space.
        grid_size: grid numbers to divide the search space.
    Return:
        a defined Searchspace object
    """
    constraints = []

    def add_subscore(obj):
        obj_elems = obj.split("_")
        if " " in obj_elems[-2]:
            lim_type = "Priority"
        elif " " in obj_elems[-1]:
            lim_type = "DoseLimit"
        else:
            raise AttributeError("objective type not realizable")
        obj_name = "_".join(obj.split("_")[:2] + [lim_type])
        obj_name = obj_name.replace(" ", "_")
        return obj_name

    def constrain_bounds(k, v, c_sample):
        min = v[0]
        max = v[1]
        for i in c_sample:
            if add_subscore(k) in i:
                c_items = i.split()
                if c_items[1] == '<=' and float(c_items[-1]) < v[1]:
                    max = float(c_items[-1])
                elif c_items[1] == '>=' and float(c_items[-1]) > v[1]:
                    min = float(c_items[-1])
            else:
                continue
        return min, max

    obj_dict = {}
    for k, v in p_dict.items():
        if grid is False:
            if not c_sample:
                param = RangeParameter(name=k,
                                       parameter_type=ParameterType.FLOAT,
                                       lower=v[0],
                                       upper=v[1])
            else:
                min, max = constrain_bounds(k, v, c_sample)
                param = RangeParameter(name=k,
                                       parameter_type=ParameterType.FLOAT,
                                       lower=min,
                                       upper=max)
            obj_dict[add_subscore(k)] = param
        else:
            values = np.linspace(v[0], v[1], grid_size)
            param = ChoiceParameter(name=k,
                                    parameter_type=ParameterType.FLOAT,
                                    sort_values=True,
                                    is_ordered=True,
                                    values=values)
            obj_dict[add_subscore(k)] = param
    if c_strs:
        for c_str in c_strs:
            constraints.append(constraint_from_str(c_str, obj_dict))
    s_p = SearchSpace(obj_dict.values(), constraints)
    return s_p

def form_key(row, col):
    """string construct helper function
    """
    if col == "Volume":
        v_str = "Volume"
        l_str = " ".join(["DoseLimit", str(row["DoseLimit"])])
        p_str = " ".join(["Priority", str(row["Priority"])])
    elif col == "DoseLimit":
        v_str = " ".join(["Volume", str(row["Volume"])])
        l_str = "DoseLimit"
        p_str = " ".join(["Priority", str(row["Priority"])])
    elif col == "Priority":
        v_str = " ".join(["Volume", str(row["Volume"])])
        l_str = " ".join(["DoseLimit", str(row["DoseLimit"])])
        p_str = "Priority"
    return "_".join([row["Structure"], row["Type"], row["limit"], v_str, l_str, p_str])


