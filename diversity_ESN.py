import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from dataset_loader import timeseries_loader, input_target_spliter
import ESN
import statsmodels.api as sm
from torch.nn.modules import Module
from sklearn.linear_model import Ridge


class Linearregressor(Module):
    def __init__(self, factor):
        super(Linearregressor, self).__init__()
        self.factor = factor

    def train(self, input_X, target_Y):
        self.regressor = Ridge(alpha=self.factor).fit(input_X, target_Y)

    def test(self, input_X):
        return self.regressor.predict(input_X)

    def extra_repr(self):
        return 'Regularization_factor={}'.format(self.factor)


class Decomposer(Module):
    def __init__(self, num_decomposer: int, sf: list):
        super(Decomposer, self).__init__()
        self.num_decomposer = num_decomposer
        self.sf = sf
        # sf means smoothing factor
        if num_decomposer == len(sf):
            self.decomposer = sm.tsa.filters.hpfilter
        else:
            raise (ValueError("The length of sf should be same with num_decomposer"))

    def extra_repr(self):
        return 'num_decomposer={}, smoothing_factor={}'.format(self.num_decomposer, self.sf)

    def forward(self, input_series):
        pending_data = None
        createVar = {"components": dict()}
        for nd in range(self.num_decomposer):
            if nd == 0:
                pending_data = input_series
            else:
                pending_data = createVar["components"]["cycle_{}".format(nd - 1)]
            temp_cycle, temp_trend = self.decomposer(pending_data, self.sf[nd])
            createVar["components"]["trend_{}".format(nd)] = temp_trend
            createVar["components"]["cycle_{}".format(nd)] = temp_cycle
        return createVar["components"]


class Encoder(Module):
    def __init__(self, in_channels: int, out_channels: int, leaking_rate_mean: float, leaking_rate_std: float,
                 in_scale_mean: float, in_scale_std: float, res_density_mean: float, res_density_std: float,
                 num_encoder: int, param_dist: str):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.leaking_rate_mean = leaking_rate_mean
        self.leaking_rate_std = leaking_rate_std
        self.in_scale_mean = in_scale_mean
        self.in_scale_std = in_scale_std
        self.res_density_mean = res_density_mean
        self.res_density_std = res_density_std
        self.num_encoder = num_encoder
        self.param_dist = param_dist

        for nd in range(self.num_encoder):
            lr = self.generate_param(self.leaking_rate_mean, self.leaking_rate_std, 0.0001, 1)
            in_scale = self.generate_param(self.in_scale_mean, self.in_scale_std, 0.1, 10)
            res_density = self.generate_param(self.res_density_mean, self.res_density_std, 0.0001, 1)
            setattr(self, 'encoder_{}'.format(nd),
                    ESN.ESN(input=in_channels, reservoir=out_channels, sr=0.95, scale_in=in_scale, leaking_rate=lr,
                            density=res_density))

    def generate_param(self, mean, std, min_val, max_val):
        if self.param_dist == 'normal':
            param = np.random.normal(mean, std)
        elif self.param_dist == 'uniform':
            param = np.random.uniform(mean - std, mean + std)
        elif self.param_dist == 'random':
            param = np.random.rand() * (max_val - min_val) + min_val
        elif self.param_dist == 'fixed':
            param = mean
        else:
            raise ValueError("Unsupported distribution type")
        return np.clip(param, min_val, max_val)


class Decoder(Module):
    def __init__(self, num_encoder: int, regularization_factor: float):
        super(Decoder, self).__init__()
        self.num_decoder = num_encoder
        self.regularization_factor = regularization_factor
        for nd in range(self.num_decoder):
            setattr(self, 'decoder_{}'.format(nd), Linearregressor(factor=self.regularization_factor))


class HPMRESN(Module):
    def __init__(self, in_channels, out_channels, num_decomposer, sf, lr_mean, lr_std, in_scale_mean, in_scale_std,
                 res_density_mean, res_density_std, regularization, param_dist):
        super(HPMRESN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_decomposer = num_decomposer
        self.num_encoder = self.num_decomposer + 1
        self.smoothing_factor = sf
        self.leaking_rate_mean = lr_mean
        self.leaking_rate_std = lr_std
        self.in_scale_mean = in_scale_mean
        self.in_scale_std = in_scale_std
        self.res_density_mean = res_density_mean
        self.res_density_std = res_density_std
        self.regularization = regularization
        self.param_dist = param_dist
        self.decomposer = Decomposer(num_decomposer=self.num_decomposer, sf=self.smoothing_factor)
        self.decoder = Decoder(num_encoder=self.num_encoder, regularization_factor=self.regularization)

    def generate_encoder(self):
        self.MR = Encoder(in_channels=self.in_channels, out_channels=self.out_channels,
                          leaking_rate_mean=self.leaking_rate_mean, leaking_rate_std=self.leaking_rate_std,
                          in_scale_mean=self.in_scale_mean, in_scale_std=self.in_scale_std,
                          res_density_mean=self.res_density_mean, res_density_std=self.res_density_std,
                          num_encoder=self.num_encoder, param_dist=self.param_dist)

    def extra_repr(self):
        return 'decomposer={}, encoder={}'.format(self.decomposer, self.encoder)

    def forward(self):
        pass


def state_transform(state, flag=1):
    if flag == 1:
        state_dim = state[0].shape[1]
        _state = torch.Tensor(len(state), state_dim)
        for col_idx in range(len(state)):
            _state[col_idx, :] = state[col_idx]
    else:
        state_dim = state[0].shape[0]
        _state = torch.Tensor(len(state), state_dim)
        for col_idx in range(len(state)):
            _state[col_idx, :] = state[col_idx]
    return _state


def evaluate_model(in_channels, out_channels, num_decomposer, sf, lr_mean, lr_std, in_scale_mean, in_scale_std,
                   res_density_mean, res_density_std, regularization, param_dist, dataset_name, validation, num_trials,
                   delay):
    model = HPMRESN(in_channels, out_channels, num_decomposer, sf, lr_mean, lr_std, in_scale_mean, in_scale_std,
                    res_density_mean, res_density_std, regularization, param_dist)
    dataset = timeseries_loader(dataset_name=dataset_name, validation=validation, delay=delay)
    components = model.decomposer(dataset.loading_data)
    components_X = dict()
    components_y = dict()
    NRMSE_validation = []
    NRMSE_test = []

    for r in tqdm(range(num_trials)):
        model.generate_encoder()
        validation_outputs = 0 if validation else None
        test_outputs = 0

        for d in range(num_decomposer):
            components_X['{}'.format(d)], components_y['{}'.format(d)] = input_target_spliter(
                components['trend_{}'.format(d)], dataset.delay)
            components_X['{}'.format(d)] = torch.Tensor(components_X['{}'.format(d)]).unsqueeze(1)
            temp_state = state_transform(getattr(model.MR, "encoder_{}".format(d))(components_X['{}'.format(d)]))
            getattr(model.decoder, "decoder_{}".format(d)).train(temp_state[dataset.train_set_index],
                                                                 components_y['{}'.format(d)][dataset.train_set_index])

            if validation:
                validation_outputs += getattr(model.decoder, "decoder_{}".format(d)).test(
                    temp_state[dataset.validation_set_index])
            test_outputs += getattr(model.decoder, "decoder_{}".format(d)).test(temp_state[dataset.test_set_index])

            if d == num_decomposer - 1:
                components_X['{}'.format(d + 1)], components_y['{}'.format(d + 1)] = input_target_spliter(
                    components['cycle_{}'.format(d)], dataset.delay)
                components_X['{}'.format(d + 1)] = torch.Tensor(components_X['{}'.format(d + 1)]).unsqueeze(1)
                temp_state = state_transform(
                    getattr(model.MR, "encoder_{}".format(d + 1))(components_X['{}'.format(d + 1)]))
                getattr(model.decoder, "decoder_{}".format(d + 1)).train(temp_state[dataset.train_set_index],
                                                                         components_y['{}'.format(d + 1)][
                                                                             dataset.train_set_index])

                if validation:
                    validation_outputs += getattr(model.decoder, "decoder_{}".format(d + 1)).test(
                        temp_state[dataset.validation_set_index])
                test_outputs += getattr(model.decoder, "decoder_{}".format(d + 1)).test(
                    temp_state[dataset.test_set_index])

        if validation:
            NRMSE_validation.append(
                mean_squared_error(y_true=dataset.target_y[dataset.validation_set_index], y_pred=validation_outputs,
                                   squared=False) / np.std(dataset.target_y[dataset.validation_set_index]))
        NRMSE_test.append(mean_squared_error(y_true=dataset.target_y[dataset.test_set_index], y_pred=test_outputs,
                                             squared=False) / np.std(dataset.target_y[dataset.test_set_index]))

    mean_validation_rmse = np.mean(np.array(NRMSE_validation)) if validation else None
    std_validation_rmse = np.std(np.array(NRMSE_validation)) if validation else None
    mean_test_rmse = np.mean(np.array(NRMSE_test))
    std_test_rmse = np.std(np.array(NRMSE_test))

    return mean_validation_rmse, std_validation_rmse, mean_test_rmse, std_test_rmse

if __name__ == '__main__':

    # Example usage
    mean_val_rmse, std_val_rmse, mean_test_rmse, std_test_rmse = evaluate_model(
        in_channels=1, out_channels=500, num_decomposer=10, sf=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        lr_mean=0.4, lr_std=0.1, in_scale_mean=1.0, in_scale_std=0.1, res_density_mean=0.1, res_density_std=0.02,
        regularization=1e-6, param_dist='normal', dataset_name='Sin_waves', validation=True, num_trials=20, delay=1
    )

    print(f'Mean RMSE on validation set: {mean_val_rmse}, Std: {std_val_rmse}')
    print(f'Mean RMSE on test set: {mean_test_rmse}, Std: {std_test_rmse}')
