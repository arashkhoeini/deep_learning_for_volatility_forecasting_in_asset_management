import torch
from torch.optim import Adam
from torch.nn import MSELoss
from data.dataset import TimeSeriesDataset
import numpy as np
import pandas as pd
import model.nets as nets
from tqdm import tqdm
from model.losses import QLIKELoss
from torch.utils.tensorboard import SummaryWriter 

class Trainer:
    def __init__(self, dataframe, configs):
        self.configs = configs
        self.model = self._init_model()
        self.dataset = self._init_data(dataframe)
        self.writer = SummaryWriter()

    
    def _init_model(self):
        if 'LSTM' in self.configs.model.arch:
            model = nets.StackedLSTM(input_size=self.configs.model.input_size, 
                        hidden_size=self.configs.model.hidden_size, 
                        output_size=self.configs.model.output_size,
                        n_layers=self.configs.model.n_layers,
                        dropout=self.configs.model.dropout)
            model = model.to(self.configs.device)
            return model
        else:
            raise NotImplementedError
        

    def _init_data(self, dataframe):
        if 'LSTM' in self.configs.model.arch:
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
        if 'LSTM' in  self.configs.model.arch:
            return self._train_lstm(optimizer, criterion)

    def _train_lstm(self, optimizer, criterion):
        
        # pretraining
        pretraining_loss = []
        progress_bar = tqdm(range(self.configs.train.p_epochs))
        for epoch in progress_bar:
            losses = []
            h_t = torch.zeros(self.configs.model.n_layers,  
                            self.configs.model.hidden_size, 
                            device=self.configs.device)
            c_t = torch.zeros(self.configs.model.n_layers,  
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
            self.writer.add_scalar('Loss/pretraining', pretraining_loss[-1], epoch)

        validation_loss = {i:[] for i in range(self.configs.model.output_size)}
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
                        predictions.append((i+k, *y_hat.cpu().tolist(), *y.cpu().tolist()))
                        for j, (y_hat_, y_) in enumerate(zip(y_hat, y)):
                            loss = criterion(y_hat_, y_)
                            validation_loss[j].append(loss.item())
                        self.model.train()
                    y_hat, (h_t, c_t) = self.model(x, h_t.detach(), c_t.detach())
                    loss = criterion(y_hat, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    seq_loss.append(loss.item())
            
                training_loss.append(loss.item())
            
            self.writer.add_scalar('Loss/training', np.array(seq_loss).mean(), i)
            progress_bar.set_postfix(loss=np.array(seq_loss).mean())   
            # progress_bar.set_postfix()
                                    
        preds_df = pd.DataFrame(predictions,)
        corrs = []
        for i in range(1, (self.configs.model.input_size//2)+1):
            j = i + self.configs.model.input_size//2
            corrs.append(preds_df[i].corr(preds_df[j]))
        self.writer.close()
        return [np.array(validation_loss[i]).mean() for i in range(len(validation_loss))], corrs
