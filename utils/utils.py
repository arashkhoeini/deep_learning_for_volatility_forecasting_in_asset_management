import os.path as osp
import os
import pandas as pd
from pathlib import Path


def mkdir(p):
    if not osp.exists(p):
        os.makedirs(p)
        print('DIR {} created'.format(p))
    return p

def read_data(configs, assets):

    if configs.model.input_size == 2:
        dataframe = pd.read_csv(assets[0])[['Var2', 'Var3']]
        return dataframe
    elif configs.model.input_size == 58:
        dataframes = {}
        for asset in assets:
            dataframes[asset] = pd.read_csv(Path(configs.data_path)/f'{asset}.csv')
        var2_columns = [df[['Var2']] for df in dataframes.values()]
        var3_columns = [df[['Var3']] for df in dataframes.values()]
        merged_var2 = pd.concat(var2_columns, axis=1)
        merged_var3 = pd.concat(var3_columns, axis=1)
    
        merged_dataframe = pd.concat([merged_var2, merged_var3], axis=1).dropna()
        return merged_dataframe
