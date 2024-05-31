import math
import torch.nn as nn
import torch

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
    def __init__(self, p_lag, n_features, future_steps, decomp_kernel_size = 7, batch_size = 8, one_layer= True):
        super(ARNet, self).__init__()
        self.one_layer = one_layer
        if one_layer: 
            self.input_trend_layer = nn.Linear(p_lag * n_features, future_steps)
            self.input_seasonal_layer = nn.Linear(p_lag * n_features, future_steps)
        else: 
            self.input_trend_layer = nn.Linear(p_lag * n_features, math.ceil(p_lag * n_features/1.5))
            self.output_trend_layer = nn.Linear(math.ceil(p_lag * n_features/1.5), future_steps)
            self.input_seasonal_layer = nn.Linear(p_lag * n_features, math.ceil(p_lag * n_features/1.5))
            self.output_seasonal_layer = nn.Linear(math.ceil(p_lag * n_features/1.5), future_steps)

        self.decomp_layer = DecompositionLayer(decomp_kernel_size, n_features)
        self.criterion = nn.MSELoss()
        self.p_lag = p_lag
        self.batch_size = batch_size
        self.n_features = n_features
        self.future_steps = future_steps

    def forward(self, input):
        input = input.float()
        input_season, input_trend = self.decomp_layer(input)
        if self.one_layer: 
            y_hat_season = self.input_seasonal_layer(input_season.reshape(self.batch_size, self.p_lag*self.n_features))
            y_hat_trend = self.input_trend_layer(input_trend.reshape(self.batch_size, self.p_lag*self.n_features))
        else: 
            y_hat_season = self.input_seasonal_layer(input_season.reshape(self.batch_size, self.p_lag*self.n_features))
            y_hat_trend = self.input_trend_layer(input_trend.reshape(self.batch_size, self.p_lag*self.n_features))
            y_hat_season = self.output_seasonal_layer(y_hat_season)
            y_hat_trend = self.output_trend_layer(y_hat_trend)
            
        return y_hat_season + y_hat_trend