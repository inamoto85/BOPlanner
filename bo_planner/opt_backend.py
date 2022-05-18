from typing import Any, Dict, Optional, Set, Iterable, Union
import numpy as np
from dataclasses import dataclass
import pyesapi as api
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.runner import Runner
from itertools import groupby
import time

@dataclass
class DVH:
    data: np.array
    total_volume: float = None
    mean_dose: float = None
    dose_unit: str = "Gy"
    volume_unit: str = "%"
    total_volume_unit: str = "cm3"
    mean_dose_unit: str = "Gy"


class OptSimulator:
    """The optimization backend, serves to communicate with the BO engine
    Args:
        plan: the plan setup to be optimized
        roi: ROI set of interested
        calc: bool control on whether to calculate the dose after the optimization.
        reopt: bool control on new opt or continue opt.
        fix_jaw: LMC optionns on fixing jaws.
        jaw_tracking: LMC option on jaw tracking.
        fixed_objectives: Dict containing fixed objectives.
        nor: bool control on whether to normalize the dose.
    """

    reopt_dict = {False: 0, True: 2}

    def __init__(self, plan: api.ExternalPlanSetup,
                 roi: Set, calc: Optional[bool] = False,
                 reopt: Optional[bool] = False,
                 fix_jaw: Optional[bool] = False,
                 jaw_tracking: Optional[bool] = True,
                 fixed_objectives: Optional[Dict] = None,
                 nor: Optional[bool] = False):
        self.plan = plan
        self.struct_roi = roi
        self.calc = calc
        self.reopt = reopt
        self.structures = self.plan.StructureSet.StructuresLot()
        self.lmc_options = api.SmartLMCOptions(fix_jaw, jaw_tracking)
        self.fixed_objectives = fixed_objectives
        self.prescription = {"struct_names": ["PTV", "PGTVp", "CTV", "GTVp"],
                             "target_covs": [95, 95, 99.9, 99.9],
                             "rx_doses": [41.8, 50.6, 41.8, 50.6]}
        self.nor = nor

    @property
    def time(self) -> float:
        return time.time()

    def set_pars(self, params: Dict, **kwargs: Any):
        """set the optimization parameters via the plan setup object
        Args:
            parms: parameter dictionary to be set
        Return:
            None
        """
        verbose = True  # kwargs.pop("verbose", False)
        if verbose:
            print("begin optimization...\n")
        opt = self.plan.OptimizationSetup
        plan_objs = opt.Objectives
        # remove all existing optimization constraints
        for obj in plan_objs:
            opt.RemoveObjective(obj)
        # TODO: add NTO weights into modifiable params?
        opt.AddAutomaticNormalTissueObjective(500)
        # adding fixed objectives first
        if self.fixed_objectives:
            for k, v in self.fixed_objectives.items():
                struct_id, lim_type, lim_dir, volume, dose, priority = parse_kw(k, v)
                self.add_obj(struct_id, lim_type,
                             lim_dir, volume, dose,
                             priority, opt)
        keys = list(params.keys())
        g_keys = groupby(keys, key=lambda x: x.split("_")[:4])  # grouped objectives for multiple variables
        for k, v in g_keys:
            struct_id, lim_type, lim_dir, volume, dose, priority = parse_kw(k, v, params)
            self.add_obj(struct_id, lim_type, lim_dir, volume, dose, priority, opt)

    def add_obj(self, struct_id: str, lim_type: str,
                lim_dir: str, volume: float,
                dose: api.DoseValue, priority: float,
                opt: api.OptimizationSetup, **kwargs) -> None:
        """add dose objectives to the `api.OptimizerSetup` object
        Args:
            struct_id: structure name in the roi dict.
            lim_type: str, objective type("Point": DVH-type objectives, "Mean": Mean dose objective)
            lim_dir: str, limit direction("lower" or "upper")
            volume: float, volume value on the DVH-type objectives
            dose: float, dose limit value on the objectives
            opt - api.OptimizationSetup, optimizer interface in PyESAPI
        Return:
            None
        """
        # parse structure and objectives
        verbose = True  # verbose = kwargs.pop("verbose", None)
        lim_dict = {"lower": api.OptimizationObjectiveOperator.Lower,
                    "upper": api.OptimizationObjectiveOperator.Upper}
        # find structure according to the structure ID
        struct = self.plan.StructureSet.StructuresLot(struct_id)
        if lim_type == "Point":
            if verbose:
                print("{} {} d{} set at {} Gy with priority {}".format(struct.Id,
                                                                       lim_dir,
                                                                       volume,
                                                                       dose.Dose,
                                                                       priority
                                                                       ))
            opt.AddPointObjective(structure=struct, objectiveOperator=lim_dict[lim_dir],
                                  dose=dose, volume=volume, priority=priority)
        elif lim_type == "Mean":
            if verbose:
                print("{} {} d_mean set at {} Gy with priority {}".format(struct.Id,
                                                                          lim_dir,
                                                                          dose.Dose,
                                                                          priority))
            opt.AddMeanDoseObjective(structure=struct,
                                     dose=dose,
                                     priority=priority)

    def run_trial(self, trial: BaseTrial):
        """optimize and register the trial with current input
        """
        self.curr_sim_start_time = self.time
        for arm in trial.arms:
            self.set_pars(arm.parameters)
            opt_res = self.plan.Optimize(api.OptimizationOptionsIMRT(1500, self.reopt_dict[self.reopt],
                                                                     1, "MLC120"))
            # retrieve relevant parameters
            if self.calc is True:
                self.calc_plan()
            # dose normalization
                if self.nor is True:
                    self.normalize()
                    print('Final dose normalization is ok !')
            arm.dvhs = self.get_dvh(opt_res, self.struct_roi)

    def calc_plan(self, max_trial=5):
        """
        calculate the plan after the optimization.
        Args:
            max_trial: maximum dose calculation trials to circumvent possible plan dose calculation failures 
            (16.1 related Eclipse bug).
        Return:
            leaf sequencing state, calculation state
        """
        print("calculating dose")
        fail_num = 0
        lmc_res = self.plan.CalculateLeafMotions(self.lmc_options)
        calc_res = self.plan.CalculateDose()
        while calc_res.Success is False:
            fail_num += 1
            if fail_num > max_trial:
                raise ValueError("failed too much times, exiting.")
            print(f"Plan dose calculation failed {fail_num} times, retrying")
            calc_res = self.plan.CalculateDose()
        return lmc_res.Success, calc_res.Success

    def stop_trial(self, trial: BaseTrial) -> None:
        if trial.status is not TrialStatus.RUNNING:
            print(f"Trial {trial.index} is not currently running (has status"
                  f"{trial.index} and cannot be stopped.")
        else:
            try:
                trial.runtime = self.time - self.curr_sim_start_time
                print(f"Trial {trial.index} took {trial.runtime} s")
            except AttributeError:
                pass

    def get_dvh(self, opt_res, roi_struct):
        """
        extract dvh list from the optimized result
        params:
        ===========
        opt_res: optimization result
        roi_struct: structure set of which dvh is to be extracted
        calc: indicate whether the opt_res is a optimized result or calculated result
        """
        dvhs = {}
        # check the calculated state with self.calc
        # TODO: see if self.plan is calculated can be checked
        if self.plan.IsDoseValid and self.calc is True:
            # extract dvh from the calculated dose
            for name in roi_struct:
                struct = self.structures[name]
                dvh = self.plan.GetDVHCumulativeData(struct,
                                                     api.DoseValuePresentation.Absolute,
                                                     api.VolumePresentation.Relative,
                                                     .01)
                if dvh is None:
                    raise ValueError(f"DVH for {name} is not available")
                dvhs[name] = convert_dvh(dvh)
        else:
            for dvh in opt_res.StructureDVHs:
                if dvh.Structure.Id in roi_struct:
                    dvhs[dvh.Structure.Id] = convert_dvh(dvh)
        return dvhs

    def normalize(self):
        """normalizing plan to the prescription level
        supports multiple target ROIs
        """
        # check if plan dose is valid:
        assert self.plan.IsDoseValid, "Plan normalization needs to be performed after dose calculation, the dose calculation is invalid now."
        self.plan.PlanNormalizationValue = 100.0
        norms = []
        for i, struct in enumerate(self.prescription["struct_names"]):
            norms.append(self.norm_factor(struct,
                                          self.prescription["target_covs"][i],
                                          rx_dose=self.prescription["rx_doses"][i]))
        self.plan.PlanNormalizationValue = min(norms)

    def norm_factor(self, struct_name, target_cov, rel_dose=100,
                    rx_dose=None):
        """normalization to the prescription level in the struct_name ROI
        Args:
            plan: ExternalPlanSetup object
            struct_name: str, the plan structure
            rel_dose: relative dose level w.r.t the prescription
            rx_dose: total prescription dose in Gy
        Return:
            normalizing factor to use (in %)
        """
        norm_val = 100  # not performing any scaling
        target = self.plan.StructureSet.StructuresLot(struct_name)
        if rx_dose is None:
            rx_dose = self.plan.TotalDose.Dose
        dv = api.DoseValue(rx_dose / 100 * rel_dose, api.DoseValue.DoseUnit.Gy)
        cov = self.plan.GetVolumeAtDose(target, dv, api.VolumePresentation.Relative)
        if cov != target_cov:
            dv = self.plan.GetDoseAtVolume(target,
                                           target_cov,
                                           api.VolumePresentation.Relative,
                                           api.DoseValuePresentation.Absolute)
            norm_val = 100 * dv.Dose / (rel_dose / 100 * rx_dose)
        return norm_val


class OptRunner(Runner):
    """an agent class to communicate with the TPS simulator class and the BO engine
    The main role is to register optization input-output info related to each trial
    """
    def __init__(self, simulator: OptSimulator, calc_plan: Optional[bool] = False,
                 reopt: Optional[bool]=False, normalize: Optional[bool] = False) -> None:
        self.simulator = simulator
        self.calc_plan = calc_plan
        self.reopt = reopt
        self.normalize = normalize

    def run(self, trial: BaseTrial) -> BaseTrial:
        self.simulator.calc = self.calc_plan
        self.simulator.nor = self.normalize
        self.simulator.reopt = self.reopt
        self.simulator.run_trial(trial=trial)
        return trial

    def stop(self, trial: BaseTrial, reason: Optional[str] = None) -> Dict[str, Any]:
        self.simulator.stop_trial(trial)
        return {"reason": reason} if reason else {}


def convert_dvh(dvh):
    """helper function to convert DVH CurveData object in ESAPI into DVH dataclass object.
    Args:
        dvh: ESAPI dvh curve object to be converted.
    Return:
        dvh: DVH dataclass object.
    """
    data = np.array([[p.DoseValue.Dose, p.Volume] for p in dvh.CurveData])
    dose_unit = dvh.CurveData[0].DoseValue.UnitAsString
    if isinstance(dvh, api.OptimizerDVH):  # optimizer DVH
        volume = dvh.Structure.Volume
        mean_dose = None
    elif isinstance(dvh, api.DVHData):  # calculated DVH
        volume = dvh.Volume
        mean_dose = dvh.MeanDose.Dose
    dvh = DVH(data, dose_unit=dose_unit,
              volume_unit=dvh.CurveData[0].VolumeUnit,
              total_volume=volume,
              mean_dose=mean_dose,)
    return dvh


def parse_kw(k: str, v: Union[float, Iterable], params=None):
    """Objective parser function
    """
    if isinstance(v, float):
        struct_id, _, lim_type, lim_dir, volume, dose, priority = k.split("_")
        if len(volume.split(" ")) > 1:
            volume = float(volume.split(" ")[-1])
        else:
            volume = float(v)
        if len(dose.split(" ")) > 1:
            dose = api.DoseValue(float(dose.split(" ")[-1]), "Gy")
        else:
            dose = api.DoseValue(float(v), "Gy")
        if len(priority.split(" ")) > 1:
            priority = float(priority.split(" ")[-1])
        else:
            priority = float(v)
    else:
        if params is None:
            raise NotImplementedError("when v is iterable, the params needs to be passed in.")
        for it in v:
            struct_id, _, lim_type, lim_dir, t_volume, t_dose, t_priority = it.split("_")
            vol_ent = t_volume.split(" ")
            if len(vol_ent) > 1:
                if not vol_ent[1].startswith("["):
                    volume = float(vol_ent[1])
            else:
                volume = params[it]
            dose_ent = t_dose.split(" ")
            if len(dose_ent) > 1:
                if not dose_ent[1].startswith("["):
                    dose = api.DoseValue(float(dose_ent[1]), "Gy")
            else:
                dose = api.DoseValue(float(params[it]), "Gy")
            pri_ent = t_priority.split(" ")
            if len(pri_ent) > 1:
                if not pri_ent[1].startswith("["):
                    priority = float(pri_ent[1])
            else:
                priority = float(params[it])
    return struct_id, lim_type, lim_dir, volume, dose, priority
