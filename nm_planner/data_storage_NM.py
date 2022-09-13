import csv
import pandas as pd


class DataStorage():
    """
    class for data stoarge and output.
    """

    def __init__(self, opt_prefix, dvh_prefix, p_dict):
        self.opt_prefix = opt_prefix
        self.dvh_prefix = dvh_prefix
        self.pars_file, self.pars_writer = DataStorage.creat_file(self, p_dict)

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
        return pars_file, pars_writer

    def save_init_data(self, data, pars_random):
        for j in range(len(pars_random)):
            pars_random[j].append(data[j])
        self.pars_writer.writerows(pars_random)
        self.pars_file.flush()

    def save_opt_data(self, data, utility):
        data = list(data)
        data.append(utility)
        self.pars_writer.writerow(data)
        self.pars_file.flush()

    def save_cal_data(self, data, best_params):
        pars_reopt = list(best_params.values())
        pars_reopt.append(data.df['mean'][len(data.df['mean']) - 1])
        self.pars_writer.writerow(pars_reopt)
        self.pars_file.flush()

    def save_dvh(self, opt_dvh, filename=''):
        DVHValue = {}
        structures = opt_dvh.keys()
        for stru in structures:
            items = opt_dvh[stru].__dict__
            for item in items.keys():
                if item == 'data':
                    DVHValue['{}_Dose'.format(stru)] = items[item][:, 0]
                    DVHValue['{}_Volume'.format(stru)] = items[item][:, -1]
                else:
                    DVHValue['{}_{}'.format(stru, item)] = [items[item]]
        dvh_df = pd.DataFrame.from_dict(DVHValue, orient='index')
        dvh_df.to_csv('{}DVH_{}.csv'.format(self.dvh_prefix, filename))

    def close_pars_file(self):
        self.pars_file.close()




