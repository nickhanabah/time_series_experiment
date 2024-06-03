import math
import torch.nn as nn
import torch
import os 
import random 
import numpy as np 
from pytorch_forecasting.metrics.quantile import QuantileLoss

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

class DecompositionLayer(nn.Module):
    def __init__(self, kernel_size, n_features):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0) 
        self.n_features = n_features

    def forward(self, x):
        num_of_pads = (self.kernel_size - 1) // 2
        if self.kernel_size > self.n_features: 
            front = x[:, 0:1, :].repeat(1, num_of_pads + 1, 1)
        else: 
            front = x[:, 0:1, :].repeat(1, num_of_pads, 1)
        end = x[:, -1:, :].repeat(1, num_of_pads, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        x_trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        x_seasonal = x - x_trend
        return x_seasonal, x_trend

class ARNet(nn.Module):
    def __init__(   self, 
                    p_lag, 
                    n_continous_features,
                    n_categorial_features, 
                    future_steps, 
                    decomp_kernel_size = 7, 
                    batch_size = 8, 
                    model:str = 'minmaxlinear', 
                    optimization='mse'):
        
        super(ARNet, self).__init__()
        self.model = model
        self.optimization = optimization
        self.n_categorial_features = n_categorial_features

        if self.optimization == 'mse': 
            self.criterion = nn.MSELoss()
            
        else: 
            raise NotImplementedError

        if model == 'dlinear': 
            print('Dlinear activated')
            self.decomp_layer = DecompositionLayer(decomp_kernel_size, n_continous_features)
            self.input_trend_layer = nn.Linear(p_lag * (n_continous_features + n_categorial_features), future_steps)
            self.input_seasonal_layer = nn.Linear(p_lag * (n_continous_features + n_categorial_features), future_steps)
        
        elif model == 'rlinear': 
            print('Rlinear activated')
            self.input_layer = nn.Linear(p_lag * (n_continous_features + n_categorial_features), future_steps)
        
        elif model == 'rlmp': 
            print('RLMP activated')
            self.input_layer = nn.Linear(p_lag * (n_continous_features + n_categorial_features), p_lag * (n_continous_features + n_categorial_features))
            self.relu = nn.ReLU()
            self.output_layer = nn.Linear(p_lag * (n_continous_features + n_categorial_features), future_steps)

        else: 
            raise NotImplementedError
  
        self.criterion = nn.MSELoss()
        self.dropout = nn.Dropout(p=0.2)
        self.p_lag = p_lag
        self.batch_size = batch_size
        self.n_continous_features = n_continous_features
        self.future_steps = future_steps

    def forward(self, input):
        input = input.float()

        if self.model == 'rlinear': 
            new_input = input.reshape(self.batch_size,(self.n_continous_features + self.n_categorial_features), self.p_lag)
            continous_input = new_input[:, 0:(self.n_continous_features), :]
            categorial_input = new_input[:, self.n_continous_features:(self.n_continous_features + self.n_categorial_features), :]

            #continous_input tranformation
            mean_values = torch.mean(continous_input ,dim=2).reshape(self.batch_size,self.n_continous_features, 1)
            mean_adj_input = continous_input - mean_values
            std_values = torch.std(continous_input, dim = 2).reshape(self.batch_size,self.n_continous_features, 1)
            eps_values = torch.full((self.batch_size,self.n_continous_features, 1), 1)
            standardized_input = mean_adj_input/(std_values + eps_values)

            # put all parts together again
            standardized_input = torch.cat((standardized_input, categorial_input), 1)
            standardized_input = self.dropout(standardized_input)
            y_hat = self.input_layer(standardized_input.reshape(self.batch_size, self.p_lag*(self.n_continous_features + self.n_categorial_features)))
            rev_mean = mean_values.squeeze(2)[:,self.n_continous_features - 1].reshape(self.batch_size, 1) 
            rev_std = std_values.squeeze(2)[:,self.n_continous_features - 1].reshape(self.batch_size, 1)
            rev_eps = torch.full((self.batch_size, 1), 1)
            y_hat = y_hat * (rev_std + rev_eps) + rev_mean

        elif self.model == 'dlinear': 
            continous_input = new_input[:, 0:(self.n_continous_features), :]
            categorial_input = new_input[:, self.n_continous_features:(self.n_continous_features + self.n_categorial_features), :]
            
            #continous_input tranformation
            input_season, input_trend = self.decomp_layer(continous_input)
            
            input_season = torch.cat((input_season, categorial_input), 1)
            input_trend = torch.cat((input_trend, categorial_input), 1)
            input_season = self.dropout(input_season)
            input_trend = self.dropout(input_trend)
            y_hat_season = self.input_seasonal_layer(input_season.reshape(self.batch_size, self.p_lag*(self.n_continous_features + self.n_categorial_features)))
            y_hat_trend = self.input_trend_layer(input_trend.reshape(self.batch_size, self.p_lag*(self.n_continous_features + self.n_categorial_features))) 
            y_hat = y_hat_season + y_hat_trend
        
        elif self.model == 'rlmp': 
            new_input = input.reshape(self.batch_size,(self.n_continous_features + self.n_categorial_features), self.p_lag)
            continous_input = new_input[:, 0:(self.n_continous_features), :]
            categorial_input = new_input[:, self.n_continous_features:(self.n_continous_features + self.n_categorial_features), :]

            #continous_input tranformation
            mean_values = torch.mean(continous_input ,dim=2).reshape(self.batch_size,self.n_continous_features, 1)
            mean_adj_input = continous_input - mean_values
            std_values = torch.std(continous_input, dim = 2).reshape(self.batch_size,self.n_continous_features, 1)
            eps_values = torch.full((self.batch_size,self.n_continous_features, 1), 1)
            standardized_input = mean_adj_input/(std_values + eps_values)

            # put all parts together again
            standardized_input = torch.cat((standardized_input, categorial_input), 1)
            standardized_input = self.dropout(standardized_input)
            y_hat = self.input_layer(standardized_input.reshape(self.batch_size, self.p_lag*(self.n_continous_features + self.n_categorial_features)))

            #prediction part
            standardized_input = self.dropout(standardized_input)
            y_hat = self.relu(self.input_layer(standardized_input.reshape(self.batch_size, self.p_lag*(self.n_continous_features + self.n_categorial_features))))
            y_hat = y_hat.reshape(self.batch_size,(self.n_continous_features + self.n_categorial_features), self.p_lag) + standardized_input
            y_hat = self.output_layer(y_hat.reshape(self.batch_size, self.p_lag*(self.n_continous_features + self.n_categorial_features)))
            rev_mean = mean_values.squeeze(2)[:,self.n_continous_features - 1].reshape(self.batch_size, 1) 
            rev_std = std_values.squeeze(2)[:,self.n_continous_features - 1].reshape(self.batch_size, 1)
            rev_eps = torch.full((self.batch_size, 1), 1)
            y_hat = y_hat * (rev_std + rev_eps) + rev_mean
            
        else: 
            raise NotImplementedError
        
        return y_hat