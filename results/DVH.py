import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib; matplotlib.use('TkAgg')
from matplotlib.pyplot import MultipleLocator



def SketchBounds(path, savepath, gp_f, saas_f, clin_f, sobol_f, DVH_items):
    color = ['purple',  'yellow', 'violet',  'green','red', 'blue','darkorange']
    clin_csv = pd.read_csv(path + clin_f)
    clin_df = pd.DataFrame(clin_csv)
    saas_csv = pd.read_csv(path + saas_f)
    saas_df = pd.DataFrame(saas_csv)
    gp_csv = pd.read_csv(path + gp_f)
    gp_df = pd.DataFrame(gp_csv)
    sobol_csv = pd.read_csv(path + sobol_f)
    sobol_df = pd.DataFrame(sobol_csv)
    fig, ([ax_c, ax_r], [ax_g, ax_s]) = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
    j = 0
    for i in DVH_items:
        print(i)
        clin_mins = np.array(clin_df[i + '_Vol_means']) - np.array(clin_df[i + '_Vol_vars'])
        clin_mins = np.minimum(clin_mins, 100)
        clin_maxs = np.array(clin_df[i + '_Vol_means']) + np.array(clin_df[i + '_Vol_vars'])
        clin_maxs = np.minimum(clin_maxs, 100)
        ax_c.plot(list(clin_df[i + '_Dose']), list(clin_df[i + '_Vol_means']), color=color[j], label=i, zorder=1)
        ax_c.fill_between(list(clin_df[i + '_Dose']), clin_mins, clin_maxs, color=color[j], alpha=0.2, zorder=1)
        ax_c.xaxis.set_minor_locator(MultipleLocator(2))
        ax_c.yaxis.set_minor_locator(MultipleLocator(5))
        ax_c.set_yticklabels([0, 20, 40, 60, 80, 100], fontsize=15)
        ax_c.set_xticklabels([0,10,20,30,40,50,60], fontsize=15)
        ax_c.x_major_locator = MultipleLocator(10)
        ax_c.set_title('Clinical', fontsize=24)
        ax_c.set_ylim(0, 100)
        ax_c.set_xlim(0, 61)
        ax_c.set_ylabel('Volume (%)', fontsize=18)

        saas_mins = np.array(saas_df[i + '_Vol_means']) - np.array(saas_df[i + '_Vol_vars'])
        saas_mins = np.minimum(saas_mins, 100)
        saas_maxs = np.array(saas_df[i + '_Vol_means']) + np.array(saas_df[i + '_Vol_vars'])
        saas_maxs = np.minimum(saas_maxs, 100)
        ax_s.plot(list(saas_df[i + '_Dose']), list(saas_df[i + '_Vol_means']), color=color[j], zorder=1)
        ax_s.fill_between(list(saas_df[i + '_Dose']), saas_mins, saas_maxs, color=color[j], alpha=0.2, zorder=1)
        ax_s.x_major_locator = MultipleLocator(10)
        ax_s.xaxis.set_minor_locator(MultipleLocator(2))
        ax_s.yaxis.set_minor_locator(MultipleLocator(5))
        ax_s.set_xticklabels([0, 10, 20, 30, 40, 50, 60], fontsize=15)
        ax_s.set_yticklabels([0, 20, 40, 60, 80, 100], fontsize=15)
        ax_s.set_title('SAAS-BO', fontsize=24)
        ax_s.set_ylim(0, 100)
        ax_s.set_xlim(0, 61)
        ax_s.set_xlabel('Dose (Gy)', fontsize=18)

        gp_mins = np.array(gp_df[i + '_Vol_means']) - np.array(gp_df[i + '_Vol_vars'])
        gp_mins = np.minimum(gp_mins, 100)
        gp_maxs = np.array(gp_df[i + '_Vol_means']) + np.array(gp_df[i + '_Vol_vars'])
        gp_maxs = np.minimum(gp_maxs, 100)
        ax_g.plot(list(gp_df[i + '_Dose']), list(gp_df[i + '_Vol_means']), color=color[j], zorder=1)
        ax_g.fill_between(list(gp_df[i + '_Dose']), gp_mins, gp_maxs, color=color[j], alpha=0.2, zorder=1)
        ax_g.get_shared_x_axes().join(ax_s, ax_g)
        ax_g.x_major_locator = MultipleLocator(10)
        ax_g.xaxis.set_minor_locator(MultipleLocator(2))
        ax_g.set_xticklabels([0, 10, 20, 30, 40, 50, 60], fontsize=15)
        ax_g.yaxis.set_minor_locator(MultipleLocator(5))
        ax_g.set_yticklabels([0, 20, 40, 60, 80, 100], fontsize=15)
        ax_g.set_title('GPEI', fontsize=24)
        ax_g.set_ylim(0, 100)
        ax_g.set_xlim(0, 61)
        ax_g.set_ylabel('Volume (%)', fontsize=18)
        ax_g.set_xlabel('Dose (Gy)', fontsize=18)

        sobol_mins = np.array(sobol_df[i + '_Vol_means']) - np.array(sobol_df[i + '_Vol_vars'])
        sobol_mins = np.minimum(sobol_mins, 100)
        sobol_maxs = np.array(sobol_df[i + '_Vol_means']) + np.array(sobol_df[i + '_Vol_vars'])
        sobol_maxs = np.minimum(sobol_maxs, 100)
        ax_r.plot(list(sobol_df[i + '_Dose']), list(sobol_df[i + '_Vol_means']), color=color[j], zorder=1)
        ax_r.fill_between(list(sobol_df[i + '_Dose']), sobol_mins, sobol_maxs, color=color[j], alpha=0.2, zorder=1)
        ax_r.get_shared_x_axes().join(ax_s, ax_g)
        ax_r.x_major_locator = MultipleLocator(10)
        ax_r.xaxis.set_minor_locator(MultipleLocator(2))
        ax_r.yaxis.set_minor_locator(MultipleLocator(5))
        ax_r.set_yticklabels([0, 20, 40, 60, 80, 100], fontsize=15)
        ax_r.set_xticklabels([0, 10, 20, 30, 40, 50, 60], fontsize=15)
        ax_r.set_title('Random', fontsize=24)
        ax_r.set_ylim(0, 100)
        ax_r.set_xlim(0, 61)
        j += 1

    ax_c.axvline(x=41.8, linestyle='--', linewidth=0.55, color='dimgrey', alpha=0.9, zorder=2)
    ax_c.axvline(x=50.6, linestyle='--', linewidth=0.55, color='dimgrey', alpha=0.9, zorder=2)
    ax_s.axvline(x=41.8, linestyle='--', linewidth=0.55, color='dimgrey', alpha=0.9, zorder=2)
    ax_s.axvline(x=50.6, linestyle='--', linewidth=0.55, color='dimgrey', alpha=0.9, zorder=2)
    ax_g.axvline(x=41.8, linestyle='--', linewidth=0.55, color='dimgrey', alpha=0.9, zorder=2)
    ax_g.axvline(x=50.6, linestyle='--', linewidth=0.55, color='dimgrey', alpha=0.9, zorder=2)
    ax_r.axvline(x=50.6, linestyle='--', linewidth=0.55, color='dimgrey', alpha=0.9, zorder=2)
    ax_r.axvline(x=41.8, linestyle='--', linewidth=0.55, color='dimgrey', alpha=0.9, zorder=2)
    ax_c.legend(loc=9, bbox_to_anchor=(0.4, 1), frameon=False, prop={'size': 13})
    plt.tight_layout()
    plt.savefig(savepath + "dvh_comparison.pdf", bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    DVH_items = ['PGTV', 'GTV', 'PTV', 'CTV', 'Femoral Head', 'Bladder', 'Body']
    path = "./data/dvh_comparison/"
    savepath = './fig/'
    file_gp = 'GP_DVH_bounds.csv'
    file_saas = 'SAAS_DVH_bounds.csv'
    file_clin = 'Clinical_DVH_bounds.csv'
    file_sobol = 'SOBOL_DVH_bounds.csv'

    SketchBounds(path, savepath, file_gp, file_saas, file_clin, file_sobol, DVH_items)

