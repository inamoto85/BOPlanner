import pandas as pd
import matplotlib; matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
sns.set(style="ticks")

gpei_path = './data/optimal_params/gpei_pars.csv'
random_path = './data/optimal_params/random_pars.csv'
saas_path = './data/optimal_params/saas_pars.csv'

gpei_df = pd.read_csv(gpei_path)
random_df = pd.read_csv(random_path)
saas_df = pd.read_csv(saas_path)

gpei_prio_df = gpei_df.iloc[:, [i % 2 == 1 for i in range(len(gpei_df.columns))]]
gpei_dose_df = gpei_df.iloc[:, [i % 2 == 0 for i in range(len(gpei_df.columns))]]
random_prio_df = random_df.iloc[:, [i % 2 == 1 for i in range(len(random_df.columns))]]
random_dose_df = random_df.iloc[:, [i % 2 == 0 for i in range(len(random_df.columns))]]
saas_prio_df = saas_df.iloc[:, [i % 2 == 1 for i in range(len(saas_df.columns))]]
saas_dose_df = saas_df.iloc[:, [i % 2 == 0 for i in range(len(saas_df.columns))]]

gpei_dose7_df = gpei_dose_df.iloc[:, [i for i in range(7)]]
gpei_dose10_df = gpei_dose_df.iloc[:, [i for i in range(7, 17)]]
random_dose7_df = random_dose_df.iloc[:, [i for i in range(7)]]
random_dose10_df = random_dose_df.iloc[:, [i for i in range(7, 17)]]
saas_dose7_df = saas_dose_df.iloc[:, [i for i in range(7)]]
saas_dose10_df = saas_dose_df.iloc[:, [i for i in range(7, 17)]]

keys_dose7 = list(gpei_dose7_df.columns)
keys_dose7_nor = []
for k in keys_dose7:
    items = k.split('_')
    t = r'$\mathrm{D^{it0}_{it1}}$'
    t = t.replace("it0", items[0])
    t = t.replace("it1", items[1])
    t = t.replace('%', '\%')
    keys_dose7_nor.append(t)

keys_dose10 = list(gpei_dose10_df.columns)
keys_dose10_nor = []
for k in keys_dose10:
    items = k.split('_')
    t = r'$\mathrm{D^{it0}_{it1}}$'
    t = t.replace("it0", items[0])
    t = t.replace("it1", items[1])
    t = t.replace('%', '\%')
    keys_dose10_nor.append(t)

keys_prio = list(gpei_prio_df.columns)
keys_prio_nor = []
for k in keys_prio:
    items = k.split('_')
    t = r'$\mathrm{D^{it0}_{it1}}$'
    t = t.replace("it0", items[0])
    t = t.replace("it1", items[1])
    t = t.replace('%', '\%')
    keys_prio_nor.append(t)

dose7_df = pd.concat([random_dose7_df, gpei_dose7_df, saas_dose7_df], keys=('Random', 'GPEI', 'SAAS-BO'))
dose7_df = dose7_df.stack()
dose7_df = dose7_df.rename_axis(index=['Model', 'nan', 'Dose Objective'])
dose7_df = dose7_df.reset_index(level=[0, 2], name='Dose Objective (Gy)')

dose10_df = pd.concat([random_dose10_df, gpei_dose10_df, saas_dose10_df], keys=('Random', 'GPEI', 'SAAS-BO'))
dose10_df = dose10_df.stack()
dose10_df = dose10_df.rename_axis(index=['Model', 'nan', 'Dose Objective'])
dose10_df = dose10_df.reset_index(level=[0, 2], name='Dose Objective (Gy)')

prio_df = pd.concat([random_prio_df, gpei_prio_df, saas_prio_df], keys=('Random', 'GPEI', 'SAAS-BO'))
prio_df = prio_df.stack()
prio_df = prio_df.rename_axis(index=['Model', 'nan', 'WeightPars'])
prio_df = prio_df.reset_index(level=[0, 2], name='Weight')

flatui = ['#FF0033', '#FF9900', '#0066CC']
fig, ax = plt.subplots(1, 3, constrained_layout=True, figsize=(40, 40))
# Dose
sns.boxplot(data=dose7_df, y='Dose Objective', hue='Model', x='Dose Objective (Gy)', orient="h", palette=sns.color_palette(flatui), width=0.5, ax=ax[0])
ax[0].xaxis.label.set_size(45)
ax[0].yaxis.label.set_visible(False)
ax[0].set_yticklabels(keys_dose7_nor, fontsize=45)
ax[0].set_xticklabels([40, 42, 44, 46, 48, 50, 52, 54], fontsize=45)
ax[0].xaxis.set_minor_locator(MultipleLocator(1))
ax[0].set_xlim(xmin=40, xmax=54)
handles, labels = ax[0].get_legend_handles_labels()
l = ax[0].legend(loc=2, frameon=False, prop={'size': 45})
l.set_title('')

sns.boxplot(data=dose10_df, y='Dose Objective', hue='Model', x='Dose Objective (Gy)', orient="h", palette=sns.color_palette(flatui), width=0.5, ax=ax[1])
ax[1].xaxis.label.set_size(45)
ax[1].yaxis.label.set_visible(False)
ax[1].set_xticklabels([0, 10, 20, 30, 40, 50], fontsize=45)
ax[1].xaxis.set_minor_locator(MultipleLocator(5))
ax[1].set_xlim(xmin=0, xmax=52)
ax[1].set_yticklabels(keys_dose10_nor, fontsize=45)
ax[1].get_legend().remove()

# Weight
sns.boxplot(data=prio_df, y='WeightPars', hue='Model', x='Weight', orient="h", palette=sns.color_palette(flatui), width=0.5, ax=ax[2])
ax[2].xaxis.label.set_size(45)
ax[2].yaxis.label.set_visible(False)
ax[2].set_yticklabels(keys_prio_nor, fontsize=45)
ax[2].set_xticklabels([0,200,400,600,800,1000], fontsize=45)
ax[2].xaxis.set_minor_locator(MultipleLocator(100))
ax[2].get_legend().remove()

plt.savefig('./fig/distribution.pdf')
plt.show()








