import sys
sys.path.append("../") # TODO: better management to handle module namespace
import pandas as pd
import torch
import re
from ax.modelbridge.cross_validation import cross_validate, compute_diagnostics, CVResult
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
from bo_planner import opt_space
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from ax.core.observation import observations_from_data
import numpyro
from numpyro.util import enable_x64
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.ticker import AutoMinorLocator

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

file_pars = "./data/cross_validation/PQM_score_random.csv"
file_pqm = "./data/cross_validation/34D_space.csv"

def cv_data(model):
    """
    helper function to return cross validate results
    """
    cv = cross_validate(model)
    y_true = np.stack([cv_.observed.data.means for cv_ in cv]).ravel()
    y_mean = np.stack([cv_.predicted.means for cv_ in cv]).ravel()
    y_std = np.stack([np.sqrt(np.diag(cv_.predicted.covariance)) for cv_ in cv]).ravel()
    return cv, y_true, y_mean, y_std


def get_sample(df, mode="seq", start=0, train_num=100, test_num=100):
    """
    df: the total dataframe from which we select the samples
    num: number of required samples
    mode: selection mode, "random" - randomly choose `num` samples
                          "seq" - sequentially selecting samples
    """
    if mode == "random":
        train_df = df.sample(train_num)
        test_df = df.sample(test_num)
    elif mode == "seq":
        train_df = df[start:start+train_num]
        test_df = df[start+train_num:start+train_num+test_num]
    return train_df, test_df


def get_funcs(funcs):
    """
    Load constituent functions from CSV file.
    """
    func_df = pd.read_csv(funcs).astype(object)
    for index, row in func_df.iterrows():
        for col in ['DoseLimit', 'Volume', 'Priority']:
            # Tunable parameters are read in as strings '[min, max]',
            # so we need to convert them back to a list of floats.
            if isinstance(row[col], str):
                pars = [float(par) for par
                        in re.findall(r'\d+\.\d+|\d+', row[col])]
                func_df.loc[index, col] = pars if len(pars) > 1 else pars[0]
    return func_df


def get_dims(df):
    dimensions = []
    for _, row in df.iterrows():
        for par in ["DoseLimit", "Volume", "Priority"]:
            if isinstance(row[par], list):
                dimensions.append(row[par])
    return dimensions


def data_to_exp(client, x_names, xs, ys):
    """
    add data to current client.experiment
    """
    for i, y in enumerate(ys):
        parameterization, trial_index = client.attach_trial(
            parameters={name: val for name, val in zip(x_names, xs[i])})
        client.complete_trial(trial_index=trial_index, raw_data=(y, None))


def process_sample(df, mode="ax", file_pqm=None):
    """
    training_mode: `ax` or `numpyro`
    prepare the data for ax and numpyro saas-gp differently.
    """
    if mode == "ax":
        f_df, p_dict, f_dict = opt_space.parse_obj(file_pqm)
        x = df[df.columns[:-1]].values
        x_names = list(p_dict.keys())
        Y = df["Utility"].tolist()
        return x, x_names, Y
    elif mode == "numpyro":
        pqm_df = get_funcs(file_pqm)
        pars = np.array(get_dims(pqm_df)).T
        x_scaler = MinMaxScaler()
        y_scaler = StandardScaler()
        x_scaler.fit(pars)
        keys = df.keys()
        X = df[list(keys)[:-1]]
        Y = df[list(keys)[-1]]
        X_all = np.array(X)
        Y_all = np.array(Y).flatten()
        X_nor = x_scaler.transform(X_all)
        y_scaler.fit(Y_all[:, np.newaxis])
        Y_nor = y_scaler.transform(Y_all[:, np.newaxis]).flatten()
        return X_nor, Y_nor, x_scaler, y_scaler


def convert_df(cv_diag, name):
    df = pd.DataFrame(cv_diag)
    df.index = [name]
    return df


def ax_validation(ax_train_x, ax_train_y, ax_test_x, ax_test_y, ax_x_names, file_pqm=None, saas=True):
    f_df, p_dict, f_dict = opt_space.parse_obj(file_pqm)
    # parse the search space
    params = []
    result = []
    cv_res= []
    it = 0
    for k, v in p_dict.items():
        params.append(
        {
            "name": k,
            "value_type": "float",
            "type": "range",
            "bounds": v,
        })
        it += 1
    diag_res = None
    ax_client = AxClient(verbose_logging=False)
    ax_client.create_experiment(parameters=params)
    # ax model training
    data_to_exp(ax_client, ax_x_names, ax_train_x, ax_train_y)
    if saas == True:
        saas_model = Models.FULLYBAYESIAN(
            experiment=ax_client.experiment,
            data=ax_client.experiment.lookup_data(),
            num_samples=256,  # Increasing this may result in better model fits
            warmup_steps=512,  # Increasing this may result in better model fits
            gp_kernel="matern",  # "rbf" is the default in the paper, but we also support "matern"
            torch_device=tkwargs["device"],
            torch_dtype=tkwargs["dtype"],
            verbose=False,  # Set to True to print stats from MCMC
            disable_progbar=False,  # Set to False to print a progress bar from MCMC
        )
    else:
        saas_model = Models.GPEI(
            experiment=ax_client.experiment,
            data=ax_client.experiment.lookup_data()
        )
    train_obs = observations_from_data(ax_client.experiment, ax_client.experiment.lookup_data())
    # adding test points to the experiment
    new_client = AxClient(verbose_logging=False)
    new_client.create_experiment(parameters=params)
    data_to_exp(new_client, ax_x_names, ax_test_x, ax_test_y)
    test_obs = observations_from_data(new_client.experiment, new_client.experiment.lookup_data())
    test_obfs = [o.features for o in test_obs]
    test_pred = saas_model.cross_validate(cv_training_data=train_obs, cv_test_points=test_obfs)
    for i, ob in enumerate(test_obs):
        ob_mean = ob.data.means[0]
        pred_mean = test_pred[i].means[0]
        pred_std = np.sqrt(test_pred[i].covariance[0, 0])
        result.append([ob_mean, pred_mean, pred_std])
        cv_res.append(CVResult(observed=ob, predicted=test_pred[i]))
    return cv_res, result


def saas_comparison(file_pars, file_pqm, start=0):
    df = pd.read_csv(file_pars, sep=",")
    # ax saas data preparation
    train_df, test_df = get_sample(df, mode="seq", start=start)
    ax_train_x, ax_x_names, ax_train_y = process_sample(train_df,
                                                        mode="ax",
                                                        file_pqm=file_pqm)
    ax_test_x, ax_x_names, ax_test_y = process_sample(test_df,
                                                      mode="ax",
                                                      file_pqm=file_pqm)
    ax_saas_cv, ax_saas_res = ax_validation(ax_train_x, ax_train_y,
                                            ax_test_x, ax_test_y,
                                            ax_x_names, file_pqm=file_pqm)

    ax_gp_cv, ax_gp_res = ax_validation(ax_train_x, ax_train_y,
                                        ax_test_x, ax_test_y,
                                        ax_x_names, file_pqm=file_pqm, saas=False)
    comparison_plot(ax_saas_res, ax_gp_res, start, numpyro=False)
    return ax_saas_cv, ax_gp_cv

def comparison_plot(ax_res, comp_res, start, numpyro=True):
    ax_res = np.array(ax_res).T
    ax_ob, ax_pred, ax_std = ax_res
    if numpyro:
        comp_ob, comp_pred, comp_std = numpyro_res
    else:
        comp_res = np.array(comp_res).T
        comp_ob, comp_pred, comp_std = comp_res
    ax_min = np.min([np.min(ax_ob), np.min(np.array(ax_pred)-np.array(ax_std))]) - 1
    ax_max = np.max([np.max(ax_ob), np.max(np.array(ax_pred)+np.array(ax_std))]) + 1
    comp_min = np.min([np.min(comp_ob), np.min(np.array(comp_pred)-np.array(comp_std))]) - 1
    comp_max = np.max([np.max(comp_ob), np.max(np.array(comp_pred)+np.array(comp_std))]) + 1
    min_val = min(ax_min, comp_min)
    max_val = max(ax_max, comp_max)
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14, 7))
    axes[0].set_xlim(min_val, max_val)
    axes[0].set_ylim(min_val, max_val)
    axes[0].errorbar(ax_ob, ax_pred, ax_std, ecolor='black', elinewidth=0.5, marker='o', mfc='black',mec='k', mew=1, ms=5, alpha=1, capsize=3, capthick=3, linestyle="none", label="Observation")
    axes[0].plot((0, 1), (0, 1), transform=axes[0].transAxes, ls='--', c='b')
    axes[0].set_xlabel(r"True PQM", fontsize=16)
    axes[0].set_ylabel(r"Predicted PQM", fontsize=16)
    axes[0].set_title('SAAS-GP', fontsize=20)
    axes[0].xaxis.set_minor_locator(AutoMinorLocator())
    axes[0].yaxis.set_minor_locator(AutoMinorLocator())
    axes[0].tick_params(labelsize=14)
    axes[1].set_xlim(min_val, max_val)
    axes[1].set_ylim(min_val, max_val)
    axes[1].errorbar(comp_ob, comp_pred, comp_std, ecolor='black',
                     elinewidth=0.5, marker='o', mfc='black',mec='k', mew=1,
                     ms=5, alpha=1, capsize=3, capthick=3, linestyle="none", label="Observation")
    axes[1].plot((0, 1), (0, 1), transform=axes[1].transAxes, ls='--', c='b')
    axes[1].set_xlabel(r"True PQM", fontsize=16)
    axes[1].set_title('GP', fontsize=20)
    axes[1].xaxis.set_minor_locator(AutoMinorLocator())
    axes[1].yaxis.set_minor_locator(AutoMinorLocator())
    axes[1].tick_params(labelsize=14)
    plt.tight_layout()
    fig.savefig("./fig/comparison dataset {}.pdf".format(start))


if __name__ == '__main__':
    diag_res = None
    for i in range(20):
        cv_saas, cv_gp = saas_comparison(file_pars, file_pqm, start=i*100)
        diag_saas = compute_diagnostics(cv_saas)
        diag_gp = compute_diagnostics(cv_gp)
        temp_df = convert_df(diag_saas, "SAAS")
        if diag_res is None:
            diag_res = temp_df
        else:
            diag_res = diag_res.append(temp_df)
        temp_df = convert_df(diag_gp, "GP")
        diag_res = diag_res.append(temp_df)
    diag_res.to_csv("./res/cv_diagnostics.csv")

