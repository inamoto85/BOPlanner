import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib; matplotlib.use('Agg')
saas_uti_fn = "./data/pqm_progress/utility_saas.csv"
gp_uti_fn = "./data/pqm_progress/utility_gpei.csv"
sobol_uti_fn = "./data/pqm_progress/utility_random.csv"
clinical_uti_fn = "./data/pqm_progress/utility_clinical.csv"

saas_df = pd.read_csv(saas_uti_fn)
gp_df = pd.read_csv(gp_uti_fn)
sobol_df = pd.read_csv(sobol_uti_fn)
clinical_df = pd.read_csv(clinical_uti_fn)
saas_array = saas_df[saas_df.columns[1:-1]].values
gp_array = gp_df[gp_df.columns[1:-1]].values
sobol_array = sobol_df[sobol_df.columns[1:-1]].values
clinical_array = clinical_df[clinical_df.columns[1]].values

saas_mean = saas_array.mean(axis=0)
saas_std = saas_array.std(axis=0)
sobol_mean = sobol_array.mean(axis=0)
sobol_std = sobol_array.std(axis=0)
gp_mean = gp_array.mean(axis=0)
gp_std = gp_array.std(axis=0)
clinical_mean = clinical_array.mean()
clinical_std = clinical_array.std()

iters = [i+1 for i in range(len(gp_mean))]
clinical_mean = np.array([clinical_mean] * len(iters))
clinical_std = np.array([clinical_std] * len(iters))

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.plot(iters, saas_mean, label="SAAS-BO", color='red')
ax.fill_between(iters, saas_mean-saas_std, saas_mean+saas_std, alpha=0.1, color='red')

ax.plot(iters, gp_mean, label="GPEI", color='blue')
ax.fill_between(iters, gp_mean-gp_std, gp_mean+gp_std, alpha=0.1, color='blue')

ax.plot(iters, sobol_mean, label="Random", color='mediumseagreen')
ax.fill_between(iters, sobol_mean-sobol_std, sobol_mean+sobol_std, alpha=0.055, color='mediumseagreen')

ax.plot(iters, clinical_mean, label="Clinical", linewidth=1, color="k", alpha=0.5, linestyle="--")
ax.fill_between(iters, clinical_mean-clinical_std, clinical_mean+clinical_std, facecolor="k", alpha=0.05)

fig.suptitle("PQM score progress", fontsize=20)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.set_ylim((-7.5, 2))
ax.set_xlim((-0, 120))
ax.set_ylabel("PQM score", fontsize=16)
ax.set_xlabel("Iteration", fontsize=16)
plt.legend(frameon=False, fontsize=16, loc="lower right")
plt.savefig("./fig/progress_with_constraint.pdf")
