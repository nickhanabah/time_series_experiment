import torch.nn as nn
import torch
import os
import random
import numpy as np

def normal_loss(normal_dist, y):
    neg_log_likelihood = -normal_dist.log_prob(y)
    return torch.mean(neg_log_likelihood)


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
    def __init__(
        self,
        p_lag: int,
        n_continous_features: int,
        n_categorial_features: int,
        future_steps: int,
        decomp_kernel_size: int = 7,
        batch_size: int = 8,
        model: str = "rlinear",
        optimization: str = "mse",
        modelling_task: str = "univariate",
        density: bool = False,
        depth:str = 'shallow'
    ):

        super(ARNet, self).__init__()
        self.model = model
        self.optimization = optimization
        self.n_categorial_features = n_categorial_features
        self.modelling_task = modelling_task
        self.density = density
        self.depth = depth

        if self.modelling_task == "univariate":
            print("Univatiate modelling")
            self.inflation_factor = 1
            print(f"inflation factor = {self.inflation_factor}")

        elif self.modelling_task == "multivariate":
            print("Multivariate modelling")
            self.inflation_factor = n_continous_features
            print(f"inflation factor = {self.inflation_factor}")

        else:
            raise NotImplementedError

        if self.optimization == "mse":
            self.criterion = nn.MSELoss()

        else:
            raise NotImplementedError

        if model == "dlinear":
            print("Dlinear activated")
            self.decomp_layer = DecompositionLayer(
                decomp_kernel_size, n_continous_features
            )
            if self.density:
                print("Density to be estimated")
                self.mu_trend_layer = nn.Linear(
                    p_lag * (n_continous_features + n_categorial_features), 1
                )
                self.mu_seasonal_layer = nn.Linear(
                    p_lag * (n_continous_features + n_categorial_features), 1
                )
                self.std_trend_layer = nn.Linear(
                    p_lag * (n_continous_features + n_categorial_features), 1
                )
                self.std_seasonal_layer = nn.Linear(
                    p_lag * (n_continous_features + n_categorial_features), 1
                )
            else:
                print("Points to be estimated")
                if depth == 'deep': 
                    print("With a deep network")
                    self.relu = nn.ReLU()
                    self.input_trend_1layer = nn.Linear(
                        p_lag * (n_continous_features + n_categorial_features),
                        p_lag * (n_continous_features + n_categorial_features),
                    )
                    self.input_seasonal_1layer = nn.Linear(
                        p_lag * (n_continous_features + n_categorial_features),
                        p_lag * (n_continous_features + n_categorial_features),
                    )
                    self.input_trend_2layer = nn.Linear(
                        p_lag * (n_continous_features + n_categorial_features),
                        p_lag * (n_continous_features + n_categorial_features),
                    )
                    self.input_seasonal_2layer = nn.Linear(
                        p_lag * (n_continous_features + n_categorial_features),
                        p_lag * (n_continous_features + n_categorial_features),
                    )
                    self.input_trend_3layer = nn.Linear(
                        p_lag * (n_continous_features + n_categorial_features),
                        p_lag * (n_continous_features + n_categorial_features),
                    )
                    self.input_seasonal_3layer = nn.Linear(
                        p_lag * (n_continous_features + n_categorial_features),
                        p_lag * (n_continous_features + n_categorial_features),
                    )
                    self.input_trend_4layer = nn.Linear(
                        p_lag * (n_continous_features + n_categorial_features),
                        future_steps * self.inflation_factor,
                    )
                    self.input_seasonal_4layer = nn.Linear(
                        p_lag * (n_continous_features + n_categorial_features),
                        future_steps * self.inflation_factor,
                    )
                elif depth == 'shallow': 
                    print("With a shallow network")
                    self.input_trend_layer = nn.Linear(
                        p_lag * (n_continous_features + n_categorial_features),
                        future_steps * self.inflation_factor,
                    )
                    self.input_seasonal_layer = nn.Linear(
                        p_lag * (n_continous_features + n_categorial_features),
                        future_steps * self.inflation_factor,
                    )
                else: 
                    raise NotImplementedError

        elif model == "rlinear":
            print("Rlinear activated")
            # if self.density:
            #    print('Density to be estimated')
            #    self.mu_layer = nn.Linear(p_lag * (n_continous_features + n_categorial_features), 1)
            #    self.std_layer = nn.Linear(p_lag * (n_continous_features + n_categorial_features), 1)
            # else:
            print("Points to be estimated")
            self.input_layer = nn.Linear(
                p_lag * (n_continous_features + n_categorial_features),
                future_steps * self.inflation_factor,
            )

        elif model == "rmlp":
            print("RMLP activated")
            self.input_layer = nn.Linear(
                p_lag * (n_continous_features + n_categorial_features),
                p_lag * (n_continous_features + n_categorial_features),
            )
            self.relu = nn.ReLU()
            # if self.density:
            #    print('Density to be estimated')
            #    self.mu_layer = nn.Linear(p_lag * (n_continous_features + n_categorial_features), 1)
            #    self.std_layer = nn.Linear(p_lag * (n_continous_features + n_categorial_features), 1)
            # else:
            print("Points to be estimated")
            self.output_layer = nn.Linear(
                p_lag * (n_continous_features + n_categorial_features),
                future_steps * self.inflation_factor,
            )

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
        new_input = input.reshape(
            self.batch_size,
            (self.n_continous_features + self.n_categorial_features),
            self.p_lag,
        )

        if self.model == "rlinear":
            continous_input = new_input[:, 0 : (self.n_continous_features), :]
            categorial_input = new_input[
                :,
                self.n_continous_features : (
                    self.n_continous_features + self.n_categorial_features
                ),
                :,
            ]

            # continous_input tranformation
            mean_values = torch.mean(continous_input, dim=2).reshape(
                self.batch_size, self.n_continous_features, 1
            )
            mean_adj_input = continous_input - mean_values
            std_values = torch.std(continous_input, dim=2).reshape(
                self.batch_size, self.n_continous_features, 1
            )
            eps_values = torch.full((self.batch_size, self.n_continous_features, 1), 1)
            standardized_input = mean_adj_input / (std_values + eps_values)

            # put all parts together again
            standardized_input = torch.cat((standardized_input, categorial_input), 1)
            standardized_input = self.dropout(standardized_input)

            y_hat = self.input_layer(
                standardized_input.reshape(
                    self.batch_size,
                    self.p_lag
                    * (self.n_continous_features + self.n_categorial_features),
                )
            )

            if self.modelling_task == "univariate":
                rev_mean = mean_values.squeeze(2)[
                    :, self.n_continous_features - 1
                ].reshape(self.batch_size, 1)
                rev_std = std_values.squeeze(2)[
                    :, self.n_continous_features - 1
                ].reshape(self.batch_size, 1)
                rev_eps = torch.full((self.batch_size, 1), 1)
            elif self.modelling_task == "multivariate":
                rev_mean_l = []
                for tensor in mean_values.reshape(
                    self.batch_size, self.n_continous_features
                ):
                    [
                        rev_mean_l.append(
                            torch.full((self.future_steps, 1), i.item()).reshape(
                                self.future_steps
                            )
                        )
                        for i in tensor
                    ]
                rev_mean = torch.cat(rev_mean_l).reshape(
                    self.batch_size, self.n_continous_features * self.future_steps
                )
                rev_std_l = []
                for tensor in std_values.reshape(
                    self.batch_size, self.n_continous_features
                ):
                    [
                        rev_std_l.append(
                            torch.full((self.future_steps, 1), i.item()).reshape(
                                self.future_steps
                            )
                        )
                        for i in tensor
                    ]
                rev_std = torch.cat(rev_std_l).reshape(
                    self.batch_size, self.n_continous_features * self.future_steps
                )
                rev_eps = torch.full(
                    (self.batch_size, self.n_continous_features * self.future_steps), 1
                )
            else:
                NotImplementedError

            y_hat = y_hat * (rev_std + rev_eps) + rev_mean

        elif self.model == "dlinear":
            continous_input = new_input[:, 0 : (self.n_continous_features), :]
            categorial_input = new_input[
                :,
                self.n_continous_features : (
                    self.n_continous_features + self.n_categorial_features
                ),
                :,
            ]

            # continous_input tranformation
            input_season, input_trend = self.decomp_layer(continous_input)
            input_season = torch.cat((input_season, categorial_input), 1)
            input_trend = torch.cat((input_trend, categorial_input), 1)
            input_season = self.dropout(input_season)
            input_trend = self.dropout(input_trend)

            if self.density:
                mu_trend = self.mu_trend_layer(
                    input_trend.reshape(
                        self.batch_size,
                        self.p_lag
                        * (self.n_continous_features + self.n_categorial_features),
                    )
                )
                std_trend = self.std_trend_layer(
                    input_trend.reshape(
                        self.batch_size,
                        self.p_lag
                        * (self.n_continous_features + self.n_categorial_features),
                    )
                )
                mu_season = self.mu_seasonal_layer(
                    input_season.reshape(
                        self.batch_size,
                        self.p_lag
                        * (self.n_continous_features + self.n_categorial_features),
                    )
                )
                std_season = self.std_seasonal_layer(
                    input_season.reshape(
                        self.batch_size,
                        self.p_lag
                        * (self.n_continous_features + self.n_categorial_features),
                    )
                )
                self.sofplus = torch.nn.Softplus()
            else:
                if self.depth == 'deep': 
                    input_season_reshaped = input_season.reshape(
                            self.batch_size,
                            self.p_lag
                            * (self.n_continous_features + self.n_categorial_features),
                        )
                    
                    input_trend_reshaped = input_trend.reshape(
                            self.batch_size,
                            self.p_lag
                            * (self.n_continous_features + self.n_categorial_features),
                        )
                    y_hat_season = self.input_seasonal_1layer(
                        input_season_reshaped
                    )
                    y_hat_season = self.relu(self.input_seasonal_2layer(y_hat_season)) + input_season_reshaped
                    y_hat_season = self.relu(self.input_seasonal_3layer(y_hat_season)) + input_season_reshaped
                    y_hat_season = self.relu(self.input_seasonal_4layer(y_hat_season))

                    y_hat_trend = self.input_trend_1layer(
                        input_trend_reshaped
                    )
                    y_hat_trend = self.relu(self.input_trend_2layer(y_hat_trend)) + input_trend_reshaped
                    y_hat_trend = self.relu(self.input_trend_3layer(y_hat_trend)) + input_trend_reshaped
                    y_hat_trend = self.relu(self.input_trend_4layer(y_hat_trend))
                elif self.depth == 'shallow': 
                    y_hat_season = self.input_seasonal_layer(
                        input_season.reshape(
                            self.batch_size,
                            self.p_lag
                            * (self.n_continous_features + self.n_categorial_features),
                        )
                    )
                    y_hat_trend = self.input_trend_layer(
                        input_trend.reshape(
                            self.batch_size,
                            self.p_lag
                            * (self.n_continous_features + self.n_categorial_features),
                        )
                    )
                else: 
                    raise NotImplementedError
                y_hat = y_hat_season + y_hat_trend

        elif self.model == "rmlp":
            continous_input = new_input[:, 0 : (self.n_continous_features), :]
            categorial_input = new_input[
                :,
                self.n_continous_features : (
                    self.n_continous_features + self.n_categorial_features
                ),
                :,
            ]

            # continous_input tranformation
            mean_values = torch.mean(continous_input, dim=2).reshape(
                self.batch_size, self.n_continous_features, 1
            )
            mean_adj_input = continous_input - mean_values
            std_values = torch.std(continous_input, dim=2).reshape(
                self.batch_size, self.n_continous_features, 1
            )
            eps_values = torch.full((self.batch_size, self.n_continous_features, 1), 1)
            standardized_input = mean_adj_input / (std_values + eps_values)

            # put all parts together again
            standardized_input = torch.cat((standardized_input, categorial_input), 1)
            standardized_input = self.dropout(standardized_input)
            y_hat = self.input_layer(
                standardized_input.reshape(
                    self.batch_size,
                    self.p_lag
                    * (self.n_continous_features + self.n_categorial_features),
                )
            )

            # prediction part
            standardized_input = self.dropout(standardized_input)
            y_hat = self.relu(
                self.input_layer(
                    standardized_input.reshape(
                        self.batch_size,
                        self.p_lag
                        * (self.n_continous_features + self.n_categorial_features),
                    )
                )
            )
            y_hat = (
                y_hat.reshape(
                    self.batch_size,
                    (self.n_continous_features + self.n_categorial_features),
                    self.p_lag,
                )
                + standardized_input
            )
            y_hat = self.output_layer(
                y_hat.reshape(
                    self.batch_size,
                    self.p_lag
                    * (self.n_continous_features + self.n_categorial_features),
                )
            )

            if self.modelling_task == "univariate":
                rev_mean = mean_values.squeeze(2)[
                    :, self.n_continous_features - 1
                ].reshape(self.batch_size, 1)
                rev_std = std_values.squeeze(2)[
                    :, self.n_continous_features - 1
                ].reshape(self.batch_size, 1)
                rev_eps = torch.full((self.batch_size, 1), 1)
            elif self.modelling_task == "multivariate":
                rev_mean_l = []
                for tensor in mean_values.reshape(
                    self.batch_size, self.n_continous_features
                ):
                    [
                        rev_mean_l.append(
                            torch.full((self.future_steps, 1), i.item()).reshape(
                                self.future_steps
                            )
                        )
                        for i in tensor
                    ]
                rev_mean = torch.cat(rev_mean_l).reshape(
                    self.batch_size, self.n_continous_features * self.future_steps
                )
                rev_std_l = []
                for tensor in std_values.reshape(
                    self.batch_size, self.n_continous_features
                ):
                    [
                        rev_std_l.append(
                            torch.full((self.future_steps, 1), i.item()).reshape(
                                self.future_steps
                            )
                        )
                        for i in tensor
                    ]
                rev_std = torch.cat(rev_std_l).reshape(
                    self.batch_size, self.n_continous_features * self.future_steps
                )
                rev_eps = torch.full(
                    (self.batch_size, self.n_continous_features * self.future_steps), 1
                )
            else:
                NotImplementedError

            y_hat = y_hat * (rev_std + rev_eps) + rev_mean

        else:
            raise NotImplementedError

        if self.density and self.model == "dlinear":
            normal_object = torch.distributions.normal.Normal(
                (mu_trend + mu_season),
                (self.sofplus(std_trend) + self.sofplus(std_season)),
            )
            return normal_object
        else:
            return y_hat


class Gating(nn.Module):
    def __init__(self, input_dim,
                 num_experts, dropout_rate=0.1):
        super(Gating, self).__init__()

        # Layers
        self.layer1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(128, 256)
        self.leaky_relu1 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(256, 128)
        self.leaky_relu2 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layer4 = nn.Linear(128, num_experts)

    def forward(self, x):
        x = x.float()
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.leaky_relu1(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.leaky_relu2(x)
        x = self.dropout3(x)

        return torch.softmax(self.layer4(x), dim=1)


class MoE(nn.Module):
    def __init__(self, trained_experts:list[ARNet]):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList(trained_experts)
        self.criterion = nn.MSELoss()

        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False

        num_experts = len(trained_experts)
        input_dim = self.experts[0].p_lag * (self.experts[0].n_continous_features + self.experts[0].n_categorial_features)
        self.gating = Gating(input_dim, num_experts)

    def forward(self, x):
        #print('weights')
        weights = self.gating(x)
        #print(weights)
        #print('----------------------------')
        #print('outputs')
        outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=2)
        print(outputs.shape)
        weights = torch.reshape(weights, outputs.shape())
        print(weights.shape)
        #weights = weights.unsqueeze(1).expand_as(outputs)
        return torch.sum(outputs * weights, dim=2)