from configs import init_configs
import sys
import pandas as pd
from pathlib import Path
from trainer import Trainer
from utils import mkdir
import json
import os
from datetime import datetime
from utils import read_data


def main_lstm1(configs):
    results = []
    data_path = Path(configs.data_path)

    for asset in data_path.iterdir():
        if asset.name.endswith('csv'):

            dataframe = read_data(configs, [asset])
            
            trainer = Trainer(dataframe, configs)
            val_loss, corr = trainer.train()
            results.append((asset.name[:-4], val_loss[0], corr[0]))
    
    results_df = pd.DataFrame(results, columns=['Asset', configs.train.criterion, 'Pearson'])
    results_df.to_csv(output_dir/'results.csv')

def main_lstmn(configs):
    
    
    results = []
    data_path = Path(configs.data_path)

    asset_names = sorted([asset.name[:-4] for asset in data_path.iterdir() if asset.name.endswith('csv')])
    dataframe = read_data(configs, asset_names)
    
    trainer = Trainer(dataframe, configs)
    val_losses, corrs = trainer.train()
    for i, asset in enumerate(asset_names):
        results.append((asset, val_losses[i], corrs[i]))

    results_df = pd.DataFrame(results, columns=['Asset', configs.train.criterion, 'Pearson'])
    results_df.to_csv(output_dir/'results.csv')



if __name__ == '__main__':

    configs = init_configs.init_config('configs/configs.yml', sys.argv[1:])
    output_dir = os.path.join('output', datetime.now().strftime("%m-%d-%H-%M"))
    output_dir = mkdir(output_dir)
    configs['output_dir'] = output_dir
    output_dir = Path(output_dir)
    with open(output_dir/'configs.json', 'w') as f:
        json.dump(configs, f)

    if configs.model.arch == 'LSTM1':
        if configs.dataset == 'DJI500':
            configs.data_path = 'database/DJI500'
            configs.model.input_size = 2
            configs.model.output_size = 1
            main_lstm1(configs)
        else:
            raise ValueError('Invalid dataset')
    elif configs.model.arch == 'LSTMn':
        if configs.dataset == 'DJI500':
            configs.data_path = 'database/DJI500'
            configs.model.input_size = 58
            configs.model.output_size = 29
            main_lstmn(configs)
        else:
            raise ValueError('Invalid dataset')
    else:
        raise ValueError('Invalid model architecture')
    