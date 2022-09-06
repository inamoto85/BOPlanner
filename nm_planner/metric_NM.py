import numpy as np
import pandas as pd
import pyesapi as api
from typing import Optional, Any, Tuple, Dict
from nm_planner.opt_backend_NM import OptSimulator
from nm_planner.data_storage_NM import DataStorage


class PlanQualityMetric():
    """
    Ax metric class boilerplate for plan quality readout
    """
    def __init__(self, patient_id,
                 plan,
                 struct_roi,
                 goal_df: pd.DataFrame,
                 p_dict: Dict,
                 f_dict: Dict,
                 opt_prefix,
                 dvh_prefix
                 ):
        self.patient_id = patient_id
        self.plan = plan
        self.struct_roi = struct_roi
        self.goal_df = goal_df
        self.d_p = 41.8  # hard coded prescription dose for now
        self.base = 2.0
        self.p_dict = p_dict
        self.f_dict = f_dict
        self.pars_record = []
        self.pqm_score_record = []
        self.opt_prefix = opt_prefix
        self.dvh_prefix = dvh_prefix
        self.save_data = DataStorage(opt_prefix, dvh_prefix, self.p_dict)

    def _evaluate(self, dvh) -> float:
        """calculate and return the composite PQM score
        """
        results = self.get_results(dvh)
        weight = 1 / (self.base ** (np.array(self.goal_df["Priority"] - 1)))
        total_weight = np.sum(weight)
        utility = 0
        for index, row in self.goal_df.iterrows():
            value = results[row["Structure"]]
            term = weight[index] * get_term(value, row)
            utility += term
        utility = utility / total_weight
        return utility

    def get_results(self, dvhs):
        result = {}
        for _, row in self.goal_df.iterrows():
            result[row["Structure"]] = self.get_value(dvhs, row, d_p=self.d_p)
        return result

    @staticmethod
    def get_value(dvhs, row, **kwargs):
        """generic readout results for the DVH
        currently supported scoring types: dvp, vdp, mean_dose
        Args:
            dvhs: dvh lists
        Return:
            one PQM score term
        """
        d_p = kwargs.pop("d_p", None)
        dose_type = kwargs.pop("dose_type", "relative")
        struct_id = str(row["Structure"]).split("_")[0]
        dvh = dvhs[struct_id]
        if row["Type"] == "Point" and row["LimitType"] == "D":
            return dvp(dvh, float(row["Volume"]), dose_type=dose_type)
        elif row["Type"] == "Point" and row["LimitType"] == "V":
            return vdp(dvh, float(row["DoseLimit"]), row["VolType"])
        elif row["Type"] == "Mean":
            return mean_dose(dvh)
        elif row["Type"] == "Spill":
            return spill(dvh, dvhs["BODY"], float(row["Volume"]), d_p)
        elif row["Type"] == "CI":
            return conformity(dvh, dvhs["BODY"], 95.0, 41.8)
        else:
            return np.nan

    def function_PQM(self, pars):
        """
        set_pars ——> opt ——> evaluate DVH ——> PQM score
        """
        # The format of pars is converted to dict
        self.pars_record.append(pars)
        keys = list(self.p_dict.keys())
        dict_pars = {}
        k = 0
        for i in keys:
            dict_pars[i] = pars[k]
            k += 1
        # set par
        opt_simulator = OptSimulator(self.plan, self.patient_id, self.struct_roi, fixed_objectives=self.f_dict)
        opt_simulator.set_pars(dict_pars)
        # opt
        opt_dvh = opt_simulator.run()
        self.save_data.save_dvh(opt_dvh, len(self.pars_record))
        # evaluate
        utility = PlanQualityMetric._evaluate(self, opt_dvh)
        self.pqm_score_record.append(utility)
        self.save_data.save_opt_data(pars, utility)
        print('evaluation was finished. PQM score : {}'.format(utility))
        return (-1) * utility


def dvp(dvh, volume, dose_type="relative"):
    """
    dose volume parameters (dvp)
    # TODO: implement the absolute volume dvp
    """
    predicate = np.where(dvh.data[:, 1] >= volume)[0]
    if volume == 100:
        return dvh.data[predicate[-1], 0]
    else:
        x1 = dvh.data[predicate[-1]]
        x2 = dvh.data[predicate[-1] + 1]
        return (x2[0] - x1[0]) / (x2[1] - x1[1]) * (volume - x1[1]) + x1[0]

def vdp(dvh, dose, vol_type):
    """
    relative/absolute volume for a given dose from the dvh curve
    vol_type: volume output type
    currently only support absolute dose readout
    """
    predicate = np.where(dvh.data[:, 0] > dose)[0]
    if len(predicate) == 0:
        vol = 0.0
    elif len(predicate) == dvh.data.shape[0]:
        vol = dvh.data[predicate[0]]
    else:
        x2 = dvh.data[predicate[0]]
        x1 = dvh.data[predicate[0]-1]
        vol = (x2[1] - x1[1]) / (x2[0] - x1[0]) * (dose - x1[0]) + x1[1]
    if vol_type == "absolute":
        vol = dvh.total_volume * vol * 0.01
    return vol

def mean_dose(dvh):
    """Mean dose
    """
    if dvh.mean_dose is not None:
        return dvh.mean_dose
    total_vol = dvh.data[0, 1]
    vol = -1 * np.diff(dvh.data[:, 1])
    dose = np.diff(dvh.data[:, 0]) / 2 + dvh.data[:-1, 0]
    return np.sum(dose * vol) / total_vol

def spill(dvh, body_dvh, percent: float, d_p: float) -> float:
    """Dose spillage R_percent.
    Args:
        dvh: the target dvh for volume readout.
        body_dvh: body dvh for volume calculation.
        d_p: prescribed dose level.
    Return:
        R_percent spillage with `percent` prescription dose:
              R = tv / V_{percent / 100 * d_p}
    """
    tv = dvh.total_volume
    dose = percent / 100.0 * d_p
    ind = np.where(body_dvh.data[:, 0] < dose)[0][-1]
    v_percent = (body_dvh.data[ind, 1] - body_dvh.data[ind+1, 1]) / (body_dvh.data[ind, 0] - body_dvh.data[ind+1, 0]) * (dose - body_dvh.data[ind+1, 0]) + body_dvh.data[ind+1, 1]
    v_absolute = v_percent * body_dvh.total_volume / 100
    return v_absolute / tv

def conformity(dvh, body_dvh, percent: float = 95, d_p: float=None) -> float:
    """Conformity calculation
    """
    dose = percent / 100.0 * d_p # type: ignore
    ind_body = np.where(body_dvh.data[:, 0] < dose)[0][-1]
    ind = np.where(dvh.data[:, 0] < dose)[0][-1] # corner case, ind == None or ind is the last element
    v_body_percent = (body_dvh.data[ind_body, 1] - body_dvh.data[ind_body+1, 1]) / (body_dvh.data[ind_body, 0] - body_dvh.data[ind_body+1, 0]) * (dose - body_dvh.data[ind_body+1, 0]) + body_dvh.data[ind_body+1, 1]
    # percentage volume in the target
    if ind is None:
        v_percent = 100
    elif ind == len(dvh.data) - 1:
        v_percent = 0
    else:
        v_percent = (dvh.data[ind, 1] - dvh.data[ind+1, 1]) / (dvh.data[ind, 0] - dvh.data[ind+1, 0]) * (dose - dvh.data[ind+1, 0]) + dvh.data[ind+1, 1]
    v_body_absolute = body_dvh.total_volume * v_body_percent
    v_absolute = dvh.total_volume * v_percent / 100
    return v_percent * v_absolute / v_body_absolute

def homogeneity(dvh, d_p: float = None, uv_percent: float = 98, lv_percent: float = 2):
    """computes the homogeneity of within the target volume
    """
    ind_u = np.where(dvh.data[:, 1] < uv_percent)[0][0]
    ind_l = np.where(dvh.data[:, 1] < lv_percent)[0][0]
    d_uv = (dvh.data[ind_u - 1, 0] - dvh.data[ind_u, 0]) / (dvh.data[ind_u - 1, 1] - dvh.data[ind_u, 1]) * (uv_percent - dvh.data[ind_u, 1]) + dvh.data[ind_u, 0]
    d_lv = (dvh.data[ind_l - 1, 0] - dvh.data[ind_l, 0]) / (dvh.data[ind_l - 1, 1] - dvh.data[ind_l, 1]) * (lv_percent - dvh.data[ind_l, 1]) + dvh.data[ind_l, 0]
    return (d_lv - d_uv) / d_p

def get_term(value, row, **kwargs):
    pqm_type = row["PQM_type"]
    pqm_dict = {"Linear": PQM_linear,
                "Quad": PQM_nonlinear,
                "Sigmoid": PQM_sigmoid}
    kwargs["a1"] = 0.2
    kwargs["a2"] = 3  # base numbers for sigmoid calculations
    t_or_o = int(row["TorO"])
    l_type = row['limit']
    dose_limit = row['DoseLimit']
    Volume = row['Volume']
    try:
        if row['LimitType'] == 'D':
            return pqm_dict[pqm_type](value, dose_limit, l_type, t_or_o, **kwargs)
        elif row['LimitType'] == 'V':
            return pqm_dict[pqm_type](value, Volume, l_type, t_or_o, **kwargs)
    except Exception as e:
        raise e

# Utility scores
def PQM_linear(val, lev, l_type, t_or_o, **kwargs):
    """
    Args:
    t_or_o: int{0, 1}: target or OAR, 1 for target
    0 for OARs.
    """
    if l_type == "upper":
        if t_or_o == 1:
            if val <= lev:
                return float(0)
            else:
                return float(100 * (lev - val) / lev)
        elif t_or_o == 0:
            if val is None:
                return float(10)
            elif val <= lev:
                return float(10 * (lev - val) / lev)
            else:
                return float(100 * (lev - val) / lev)
    elif l_type == "lower":
        if t_or_o == 1:
            if val is None:
                return -100.
            elif val >= lev:
                return float(0)
            else:
                return float(100 * (val - lev) / lev)
        elif t_or_o == 0:
            if val is None:
                return -100
            elif val >= lev:
                return float(10 * (val - lev) / lev)
            elif val < lev:
                return float(100 * (val - lev) / lev)
    else:
        raise NotImplementedError("invalid type: {0}".format(l_type))

def PQM_nonlinear(val, lev, l_type, t_or_o, **kwargs):
    diff = 100 * (lev - val) / lev
    if l_type == "upper":
        if val <= lev:
            return float(0)
        elif val > lev:
            return float(-1 * (diff ** 2))
    elif l_type == "lower":
        if val >= lev:
            return float(0)
        elif val < lev:
            return float(-1 * (diff ** 2))
    else:
        raise NotImplementedError("invalid type: {0}".format(l_type))

def PQM_sigmoid(val, lev, l_type, t_or_o, **kwargs):
    diff = val - lev
    a1 = kwargs.pop("a1", 0.2)
    a2 = kwargs.pop("a2", 3)
    if l_type == "upper":
        if diff < 0:
            sig = 1 / (1 + np.exp(-1 * a1 * diff))
        else:
            sig = 1 / (1 + np.exp(-1 * a2 * diff))
    elif l_type == "lower":
        if diff < 0:
            sig = 1 / (1 + np.exp(a2 * diff))
        else:
            sig = 1 / (1 + np.exp(a1 * diff))
    return float(-1*sig)
