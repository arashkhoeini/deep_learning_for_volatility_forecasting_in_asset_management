import torch
from torch.optim import Adam
from torch.nn import MSELoss
from data.dataset import TimeSeriesDataset
import numpy as np
import pandas as pd
import model.nets as nets
from tqdm import tqdm
from model.losses import QLIKELoss

class Trainer:
    def __init__(self, dataframe, configs):
        self.configs = configs

        self.model = self._init_model()
        self.dataset = self._init_data(dataframe)

    
    def _init_model(self):
        if self.configs.model.arch == 'LSTM':
            model = nets.StackedLSTM(input_size=self.configs.model.input_size, 
                        hidden_size=self.configs.model.hidden_size, 
                        output_size=self.configs.model.output_size,
                        dropout=self.configs.model.dropout)
        model = model.to(self.configs.device)
        return model

    def _init_data(self, dataframe):
        if self.configs.model.arch == 'LSTM':
            dataset = TimeSeriesDataset(dataframe=dataframe, 
                                        seq_length=self.configs.train.seq_len, 
                                        normalize=True, 
                                        device=self.configs.device)
        return dataset

    def train(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.configs.train.lr)
        if self.configs.train.criterion == "MSE":
            criterion = torch.nn.MSELoss()
        elif self.configs.train.criterion == "QLIKE":
            criterion = QLIKELoss()
        if self.configs.model.arch == 'LSTM':
            return self._train_lstm(optimizer, criterion)

    def _train_lstm(self, optimizer, criterion):

        pretraining_loss = []
        # pretraining
        progress_bar = tqdm(range(self.configs.train.p_epochs))
        for epoch in progress_bar:
            losses = []
            h_t = torch.zeros(self.configs.model.input_size,  
                            self.configs.model.hidden_size, 
                            device=self.configs.device)
            c_t = torch.zeros(self.configs.model.input_size,  
                            self.configs.model.hidden_size, 
                            device=self.configs.device)
            for i in range(280):
                _, x, y = self.dataset[i]
                y_hat, (h_t, c_t) = self.model(x, h_t.detach(), c_t.detach())
                loss = criterion(y_hat, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            pretraining_loss.append(np.array(losses).mean())
            progress_bar.set_postfix(loss=pretraining_loss[-1])

        validation_loss = []
        predictions = []
        training_loss = []
        progress_bar = tqdm(range(281, len(self.dataset), 20))
        for i in progress_bar:
            seq_loss = []
            for epoch in range(20):
                for k in range(20):
                    if i+k >= len(self.dataset):
                        break
                    _, x, y =self.dataset[i+k]
                    if epoch ==0:
                        self.model.eval()
                        y_hat, (h_t, c_t) = self.model(x, h_t.detach(), c_t.detach())
                        predictions.append((i+k, y_hat.cpu().item(), y.cpu().item()))
                        loss = criterion(y_hat, y)
                        validation_loss.append(loss.item())
                        self.model.train()
                    y_hat, (h_t, c_t) = self.model(x, h_t.detach(), c_t.detach())
                    loss = criterion(y_hat, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                training_loss.append(loss.item())
                seq_loss.append(loss.item())
            progress_bar.set_postfix(seq_loss=np.array(seq_loss).mean(), val_loss=validation_loss[-1], train_loss=np.array(training_loss).mean())
                                    
        preds_df = pd.DataFrame(predictions, columns=['index', 'prediction', 'target'])
        corr = preds_df['prediction'].corr(preds_df['target'])
        return np.array(validation_loss).mean(), corr
