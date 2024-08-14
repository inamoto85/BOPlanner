import pandas as pd


def init_data(patient_id: str, n_init: int, mode="file", file="/util_sobol/pars.csv"):
    """data initialization for the BO procedure
    Args:
        patient_id: str,
        n_init: number of initial data to populate
        mode: str, file mode "file"
        file: str, file directory to look for the initialization file.
    Return:
        return an experiment object populated with an initial dataset
    """
    pars = []
    vals = []
    if mode == "file":
        sobolpath = "./opt_res/{}/".format(patient_id) + file
        sobol_odf = pd.read_csv(sobolpath)
        for i in range(n_init):
            parameters = sobol_odf.iloc[i, :-1].to_dict()
            par = list(parameters.values())
            value = sobol_odf.iloc[i, -1]
            pars.append(par)
            vals.append(value)
    return pars, vals
