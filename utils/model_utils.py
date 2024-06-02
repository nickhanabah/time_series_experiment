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
    def __init__(self, p_lag, n_features, future_steps, decomp_kernel_size = 7, batch_size = 8, layers= 1, model:str = 'minmaxlinear', optimization='mse'):
        super(ARNet, self).__init__()
        self.layers = layers
        self.model = model
        self.optimization = optimization

        if self.optimization == 'mse': 
            self.criterion = nn.MSELoss()

        elif self.optimization == 'lowerquantile': 
            self.lowerquantilecriterion = QuantileLoss(quantiles=[0.1])

        elif self.optimization == 'upperquantile': 
            self.upperquantilecriterion = QuantileLoss(quantiles=[0.9])
            
        else: 
            raise NotImplementedError

        if model == 'minmaxlinear': 
            print('MinMaxLinear activated')
            
            print('Model running on 1 Layer')
            self.input_layer = nn.Linear(p_lag * n_features, future_steps)

        elif model == 'dlinear': 
            print('Dlinear activated')
            self.decomp_layer = DecompositionLayer(decomp_kernel_size, n_features)
            if layers ==1: 
                print('Model running on 1 Layer')
                self.input_trend_layer = nn.Linear(p_lag * n_features, future_steps)
                self.input_seasonal_layer = nn.Linear(p_lag * n_features, future_steps)
            elif layers ==2: 
                print('Model running on 2 Layers')
                self.input_trend_layer = nn.Linear(p_lag * n_features, math.ceil(p_lag * n_features/1.5))
                self.output_trend_layer = nn.Linear(math.ceil(p_lag * n_features/1.5), future_steps)
                self.relu = nn.ReLU()
                self.input_seasonal_layer = nn.Linear(p_lag * n_features, math.ceil(p_lag * n_features/1.5))
                self.output_seasonal_layer = nn.Linear(math.ceil(p_lag * n_features/1.5), future_steps)
            
            elif layers ==3: 
                print('Model running on 3 Layers')
                self.input_trend_layer = nn.Linear(p_lag * n_features, math.ceil(p_lag * n_features/1.5))
                self.hidden_trend_layer = nn.Linear(math.ceil(p_lag * n_features/1.5), math.ceil(p_lag * n_features/3))
                self.output_trend_layer = nn.Linear(math.ceil(p_lag * n_features/3), future_steps)
                self.relu = nn.ReLU()
                self.input_seasonal_layer = nn.Linear(p_lag * n_features, math.ceil(p_lag * n_features/1.5))
                self.hidden_seasonal_layer = nn.Linear(math.ceil(p_lag * n_features/1.5), math.ceil(p_lag * n_features/3))
                self.output_seasonal_layer = nn.Linear(math.ceil(p_lag * n_features/3), future_steps)
        
        elif model == 'rlinear': 
            print('Rlinear activated')
            if layers ==1: 
                print('Model running on 1 Layer')
                self.input_layer = nn.Linear(p_lag * n_features, future_steps)
            elif layers ==2: 
                print('Model running on 2 Layers')
                self.input_layer = nn.Linear(p_lag * n_features, math.ceil(p_lag * n_features/1.5))
                self.relu = nn.ReLU()
                self.output_layer = nn.Linear(math.ceil(p_lag * n_features/1.5), future_steps)
            
            elif layers ==3: 
                print('Model running on 3 Layers')
                self.relu = nn.ReLU()
                self.input_layer = nn.Linear(p_lag * n_features, math.ceil(p_lag * n_features/1.5))
                self.hidden_layer = nn.Linear(math.ceil(p_lag * n_features/1.5), math.ceil(p_lag * n_features/3))
                self.output_layer = nn.Linear(math.ceil(p_lag * n_features/3), future_steps)
        
        #todo rlmp 
        elif model == 'rlmp': 
            print('RLMP activated')
            self.input_layer = nn.Linear(p_lag * n_features, p_lag * n_features)
            self.relu = nn.ReLU()
            self.output_layer = nn.Linear(p_lag * n_features, future_steps)

        else: 
            raise NotImplementedError
  
        self.criterion = nn.MSELoss()
        self.dropout = nn.Dropout(p=0.2)
        self.p_lag = p_lag
        self.batch_size = batch_size
        self.n_features = n_features
        self.future_steps = future_steps

    def forward(self, input):
        input = input.float()
        if self.model == 'minmaxlinear': 
            new_input = input.reshape(self.batch_size,self.n_features, self.p_lag)
            min_values, _ = torch.min(new_input ,dim=2, keepdim=True)
            min_values = min_values.reshape(self.batch_size,self.n_features, 1)
            max_values, _ = torch.max(new_input ,dim=2, keepdim=True)
            max_values = max_values.reshape(self.batch_size,self.n_features, 1)
            eps_values = torch.full((self.batch_size,self.n_features, 1), 1)
            scaled_input = (new_input - min_values)/(max_values + min_values + eps_values)
            scaled_input = self.dropout(scaled_input)
            y_hat = self.input_layer(scaled_input.reshape(self.batch_size, self.p_lag*self.n_features))
            rev_min = min_values.squeeze(2)[:,self.n_features - 1].reshape(self.batch_size, 1) 
            rev_max = max_values.squeeze(2)[:,self.n_features - 1].reshape(self.batch_size, 1)
            rev_eps = torch.full((self.batch_size, 1), 1)
            y_hat = y_hat * (rev_max - rev_min + rev_eps) + rev_min
            return y_hat  


        elif self.model == 'rlinear': 
            new_input = input.reshape(self.batch_size,self.n_features, self.p_lag)
            mean_values = torch.mean(new_input ,dim=2).reshape(self.batch_size,self.n_features, 1)
            mean_adj_input = new_input - mean_values
            std_values = torch.std(new_input, dim = 2).reshape(self.batch_size,self.n_features, 1)
            eps_values = torch.full((self.batch_size,self.n_features, 1), 1)
            standardized_input = mean_adj_input/(std_values + eps_values)
            standardized_input = self.dropout(standardized_input)
            if self.layers ==1: 
                y_hat = self.input_layer(standardized_input.reshape(self.batch_size, self.p_lag*self.n_features))
            elif self.layers ==2: 
                y_hat = self.relu(self.input_layer(standardized_input.reshape(self.batch_size, self.p_lag*self.n_features)))
                y_hat = self.output_layer(y_hat)
            elif self.layers ==3: 
                y_hat = self.relu(self.input_layer(standardized_input.reshape(self.batch_size, self.p_lag*self.n_features)))
                y_hat = self.relu(self.hidden_layer(y_hat))
                y_hat = self.output_layer(y_hat)
            rev_mean = mean_values.squeeze(2)[:,self.n_features - 1].reshape(self.batch_size, 1) 
            rev_std = std_values.squeeze(2)[:,self.n_features - 1].reshape(self.batch_size, 1)
            rev_eps = torch.full((self.batch_size, 1), 1)
            y_hat = y_hat * (rev_std + rev_eps) + rev_mean
            return y_hat


        elif self.model == 'dlinear': 
            input_season, input_trend = self.decomp_layer(input)
            input_season = self.dropout(input_season)
            input_trend = self.dropout(input_trend)
            if self.layers ==1: 
                y_hat_season = self.input_seasonal_layer(input_season.reshape(self.batch_size, self.p_lag*self.n_features))
                y_hat_trend = self.input_trend_layer(input_trend.reshape(self.batch_size, self.p_lag*self.n_features))
            elif self.layers ==2: 
                y_hat_season = self.relu(self.input_seasonal_layer(input_season.reshape(self.batch_size, self.p_lag*self.n_features)))
                y_hat_trend = self.relu(self.input_trend_layer(input_trend.reshape(self.batch_size, self.p_lag*self.n_features)))
                y_hat_season = self.output_seasonal_layer(y_hat_season)
                y_hat_trend = self.output_trend_layer(y_hat_trend)
            elif self.layers ==3: 
                y_hat_season = self.relu(self.input_seasonal_layer(input_season.reshape(self.batch_size, self.p_lag*self.n_features)))
                y_hat_trend = self.relu(self.input_trend_layer(input_trend.reshape(self.batch_size, self.p_lag*self.n_features)))
                y_hat_season = self.relu(self.hidden_seasonal_layer(y_hat_season))
                y_hat_trend = self.relu(self.hidden_trend_layer(y_hat_trend))
                y_hat_season = self.output_seasonal_layer(y_hat_season)
                y_hat_trend = self.output_trend_layer(y_hat_trend)  
            return y_hat_season + y_hat_trend
        
        elif self.model == 'rlmp': 
            new_input = input.reshape(self.batch_size,self.n_features, self.p_lag)
            mean_values = torch.mean(new_input ,dim=2).reshape(self.batch_size,self.n_features, 1)
            mean_adj_input = new_input - mean_values
            std_values = torch.std(new_input, dim = 2).reshape(self.batch_size,self.n_features, 1)
            eps_values = torch.full((self.batch_size,self.n_features, 1), 1)
            standardized_input = mean_adj_input/(std_values + eps_values)
            standardized_input = self.dropout(standardized_input)

            y_hat = self.relu(self.input_layer(standardized_input.reshape(self.batch_size, self.p_lag*self.n_features)))
            y_hat = y_hat.reshape(self.batch_size,self.n_features, self.p_lag) + standardized_input
            y_hat = self.output_layer(y_hat.reshape(self.batch_size, self.p_lag*self.n_features))
            rev_mean = mean_values.squeeze(2)[:,self.n_features - 1].reshape(self.batch_size, 1) 
            rev_std = std_values.squeeze(2)[:,self.n_features - 1].reshape(self.batch_size, 1)
            rev_eps = torch.full((self.batch_size, 1), 1)
            y_hat = y_hat * (rev_std + rev_eps) + rev_mean
            return y_hat
        
        else: 
            raise NotImplementedError