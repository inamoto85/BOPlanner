import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
import matplotlib.ticker as ticker
params = {"mathtext.default": "regular"}
plt.rcParams.update(params)


def box(gpei_file, saas_file, name, rank_type='median', bold_nums=5):
    gpei = pd.read_csv(gpei_file)
    saas = pd.read_csv(saas_file)
    keys = list(gpei.columns)
    keys_nor = []
    for k in keys:
        gpei[k] = 1. / gpei[k]
        saas[k] = 1. / saas[k]
        items = k.split('_')
        if items[-1] == "DoseLimit":
            t = r'$\mathrm{D^{it0}_{it1} dose}$'
        elif items[-1] == "Priority":
            t = r'$\mathrm{D^{it0}_{it1} weight}$'
        t = t.replace("it0", items[0])
        t = t.replace("it1", items[1])
        t = t.replace('%', '\%')
        print(k)
        keys_nor.append(t)

    gpei_max = gpei.median().max()
    saas_max = saas.median().max()

    # Normalize
    for k in keys:
        gpei[k] = gpei[k] / gpei_max
        saas[k] = saas[k] / saas_max

    gpei_mean = []
    saas_mean = []
    gpei_median = []
    saas_median = []
    gpei_std = []
    saas_std = []
    for k in keys:
        gpei_mean.append(gpei[k].mean())
        gpei_std.append(gpei[k].std())
        saas_mean.append(saas[k].mean())
        saas_std.append(saas[k].std())
        saas_median.append(saas[k].median())
        gpei_median.append(gpei[k].median())

    if rank_type == 'median':
        gpei_index = list(np.argsort(gpei_median))
        saas_index = list(np.argsort(saas_median))
    elif rank_type == 'mean':
        gpei_index = list(np.argsort(gpei_mean))
        saas_index = list(np.argsort(saas_mean))

    gpei_obj_rank = [keys_nor[i] for i in gpei_index]
    saas_obj_rank = [keys_nor[i] for i in saas_index]
    gpei_key_rank = [keys[i] for i in gpei_index]
    saas_key_rank = [keys[i] for i in saas_index]

    gpei_box = [list(gpei[k]) for k in gpei_key_rank]
    saas_box = [list(saas[k]) for k in saas_key_rank]

    for i in range(bold_nums):
        ind = len(saas_obj_rank) - i - 1
        saas_obj_rank[ind] = saas_obj_rank[ind].replace("mathrm", "mathbf")
        j = gpei_index.index(saas_index[ind])
        gpei_obj_rank[j] = gpei_obj_rank[j].replace("mathrm", "mathbf")

    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(25, 23))
    ax[1].boxplot(gpei_box, vert=False, showmeans=False, showfliers=False)
    ax[0].boxplot(saas_box, vert=False, showmeans=False, showfliers=False)
    ax[1].yaxis.set_major_formatter(ticker.FixedFormatter((gpei_obj_rank)))
    ax[0].yaxis.set_major_formatter(ticker.FixedFormatter((saas_obj_rank)))
    ax[1].set_yticklabels(gpei_obj_rank, fontsize=25)
    ax[0].set_yticklabels(saas_obj_rank, fontsize=25)
    ax[0].set_xticklabels([0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], fontsize=25)
    ax[1].set_xticklabels([0.6, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4], fontsize=25)
    ax[0].set_xlabel('Relative Importance', fontsize=30)
    ax[1].set_xlabel('Relative Importance', fontsize=30)
    ax[1].set_title('GPEI', fontsize=40)
    ax[0].set_title('SAAS-BO', fontsize=40)
    plt.savefig('./fig/'+name+'.pdf')
    plt.show()



if __name__ == '__main__':
    gpei_file = './data/parameter_importance/lenscale-gpei.csv'
    saas_file = './data/parameter_importance/lenscale-saas.csv'
    box(gpei_file, saas_file, 'lengthscales')

