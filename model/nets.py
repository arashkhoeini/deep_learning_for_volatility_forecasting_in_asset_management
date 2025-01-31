import torch
import torch.nn as nn


class StackedLSTM(nn.Module):
    """
    Implementation of a 2-layer stacked LSTM model with a fully connected output layer.
    Introduced in the paper: "Deep learning for volatility forecasting in asset management".
    https://link.springer.com/article/10.1007/s00500-022-07161-1
    """
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout=0.2):
        super(StackedLSTM, self).__init__()
        
        # 2-layer stacked LSTM
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = n_layers,
            batch_first =True,
            dropout = dropout,
        )
        
        # Fully connected output layer with Softplus activation
        self.fc = nn.Linear(hidden_size, output_size)
        self.softplus = nn.Softplus()
        
    def forward(self, x, h_t=None, c_t=None):
        if (h_t is None) or (c_t is None):
            lstm_out, (h_t, c_t) = self.lstm(x)
        else:
            lstm_out, (h_t, c_t) = self.lstm(x, (h_t, c_t)) 
        
        
        # Fully connected layer with Softplus activation
        output = self.softplus(self.fc(lstm_out))
        # print(output.shape)
        return output[ -1, :].view(-1), (h_t, c_t)  # Return only the last output