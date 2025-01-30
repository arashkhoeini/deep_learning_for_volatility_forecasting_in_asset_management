from configs import init_configs
import sys
import pandas as pd
from pathlib import Path
from trainer import Trainer
from utils import mkdir
import json
import os
from datetime import datetime

def main():
    
    configs = init_configs.init_config('configs/configs.yml', sys.argv[1:])

    output_dir = os.path.join('output', datetime.now().strftime("%m-%d-%H-%M"))
    output_dir = mkdir(output_dir)
    configs['output_dir'] = output_dir
    output_dir = Path(output_dir)
    with open(output_dir/'configs.json', 'w') as f:
        json.dump(configs, f)

    results = []
    data_path = Path(configs.data_path)
    for asset in data_path.iterdir():
        if asset.name.endswith('csv'):

            dataframe = pd.read_csv(asset)[['Var2', 'Var3']]
            
            trainer = Trainer(dataframe, configs)
            val_loss, corr = trainer.train()
            results.append((asset.name[:-4], val_loss, corr))
    results_df = pd.DataFrame(results, columns=['Asset', configs.train.criterion, 'Pearson'])
    results_df.to_csv(output_dir/'results.csv')



if __name__ == '__main__':
    main()