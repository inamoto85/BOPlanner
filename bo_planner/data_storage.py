import csv
import pandas as pd


class DataStorage():
    """
    class for data stoarge and output.
    """

    def __init__(self, opt_prefix, dvh_prefix, p_dict):
        self.opt_prefix = opt_prefix
        self.dvh_prefix = dvh_prefix
        self.pars_file, self.pars_writer, self.lenfile, self.lenwriter = DataStorage.creat_file(self, p_dict)

    def creat_file(self, p_dict):
        """
        Create csv file
        """
        pars_fn = self.opt_prefix + "pars.csv"
        pars_file = open(pars_fn, 'w', newline='')
        pars_writer = csv.writer(pars_file)
        pars_title = list(p_dict.keys())
        pars_title.append('Utility')
        pars_writer.writerow(pars_title)
        lenfn = self.opt_prefix + "hyperpars.csv"
        lenfile = open(lenfn, 'w', newline='')
        lenwriter = csv.writer(lenfile)
        lentitle = list(p_dict.keys())
        lenwriter.writerow(lentitle)
        return pars_file, pars_writer, lenfile, lenwriter

    def save_init_data(self, data, pars_random):
        for j in range(len(pars_random)):
            pars_random[j].append(data.df['mean'][j])
        self.pars_writer.writerows(pars_random)
        self.pars_file.flush()

    def save_opt_data(self, data, trial, batch_size, n_init, i):
        pars_bo = [list(trial.arms[i].parameters.values()) for i in range(len(trial.arms))]
        for k in range(batch_size):
            arms_name = str(n_init + i) + '_' + str(k)
            index = data.df[data.df.arm_name == arms_name].index.tolist()[0]
            pars_bo[k].append(data.df['mean'][index])
        self.pars_writer.writerows(pars_bo)
        self.pars_file.flush()

    def save_cal_data(self, data, best_params):
        pars_reopt = list(best_params.values())
        pars_reopt.append(data.df['mean'][len(data.df['mean']) - 1])
        self.pars_writer.writerow(pars_reopt)
        self.pars_file.flush()

    def save_dvh(self, arms, filename=''):
        for i in range(len(arms)):
            DVHValue = {}
            structures = arms[i].dvhs.keys()
            for stru in structures:
                items = arms[i].dvhs[stru].__dict__
                for item in items.keys():
                    if item == 'data':
                        DVHValue['{}_Dose'.format(stru)] = items[item][:, 0]
                        DVHValue['{}_Volume'.format(stru)] = items[item][:, -1]
                    else:
                        DVHValue['{}_{}'.format(stru, item)] = [items[item]]
            name = arms[i].name
            dvh_df = pd.DataFrame.from_dict(DVHValue, orient='index')
            dvh_df.to_csv('{}DVH_{}{}.csv'.format(self.dvh_prefix, name, filename))

    def save_lenscal_saas(self, model):
        median_lengthscales = model.model.model.models[0].covar_module.base_kernel.lengthscale.squeeze().median(
            axis=0).values
        hyperpars = [median_lengthscales[h].tolist() for h in range(len(median_lengthscales))]
        self.lenwriter.writerow(hyperpars)

    def save_lenscale_gpei(self, model):
        median_lengthscales = model.model.model.covar_module.base_kernel.lengthscale.squeeze().tolist()
        self.lenwriter.writerow(median_lengthscales)

    def close_lenscale_file(self):
        self.lenfile.close()

    def close_pars_file(self):
        self.pars_file.close()




