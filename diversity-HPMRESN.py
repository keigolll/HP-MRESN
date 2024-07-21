import argparse
import numpy as np
from torch.nn.modules import Module
import statsmodels.api as sm
import ESN
from sklearn.linear_model import Ridge
from dataset_loader import timeseries_loader, input_target_spliter
import torch
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt

# This is the implementation of HP-MRESN.
# You can refer to our paper for hyperparameter settings in HP-MRESN on various datasets.
# You can add PCA into HP-MRESN by yourself for building HP-MRESN(PCA) (e.g. sklearn.decomposition.PCA)
parser = argparse.ArgumentParser()
parser.add_argument('--in_channels', type=int, default=1, help='Dimension of inputs')
parser.add_argument('--out_channels', type=int, default=500, help='Dimension of embeddings')
parser.add_argument('--num_decomposer', type=int, default=10, help='Number of HP decomposers')
parser.add_argument('--sf', type=list, default=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                    help='Smoothing factor for HP decomposers')
parser.add_argument('--lr_mean', type=float, default=0.5, help='Mean of leaking rate, (0,1]')
parser.add_argument('--lr_std', type=float, default=0, help='Standard deviation of leaking rate')
parser.add_argument('--in_scale_mean', type=float, default=1.0, help='Mean of input scaling')
parser.add_argument('--in_scale_std', type=float, default=0, help='Standard deviation of input scaling')
parser.add_argument('--res_density_mean', type=float, default=0.1, help='Mean of reservoir density, (0,1]')
parser.add_argument('--res_density_std', type=float, default=0, help='Standard deviation of reservoir density')
parser.add_argument('--param_dist', type=str, default='normal', choices=['normal', 'uniform', 'random', 'fixed'],
                    help='Distribution type for parameters')
parser.add_argument('--C', type=float, default=1e-6, help='Regularization factor of the ridge regression')
parser.add_argument('--dataset', type=str, default='Sin_waves', help='Datasets used in the paper')
# We prepared datasets used in the paper with keywords: "Cardio", "Sunspot", "Bike", "Traffic", "Melbourne", "Electricity", "MGS17", "Laser", "Nosiy_MGS17", and "Noisy_Laser".
parser.add_argument('--validation', type=bool, default=True,
                    help='Validation mode or not, If not, the validation set will be merged into the training set')
parser.add_argument('--num_trials', type=int, default=1, help="Number of random trials")
parser.add_argument('--delay', type=int, default=1, help="K-step-ahead, you can choose the value of K.")
# 予測画像を出力するか
parser.add_argument('--output', type=bool, default=True, help='Output prediction images')
args = parser.parse_args()
createVar = locals()
createVar["components"] = dict()

# Root Mean Squared Error function
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

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


class Linearregressor(Module):
    def __init__(self, factor):
        super(Linearregressor, self).__init__()
        self.factor = factor

    def train(self, input_X, target_Y):
        self.regressor = Ridge(alpha=self.factor).fit(input_X, target_Y)

    def test(self, input_X):
        return self.regressor.predict(input_X)

    def extra_repr(self):
        return 'Regularization_factor={}'.format(
            self.factor)


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
        return 'num_decomposer={}, smoothing_factor={}'.format(
            self.num_decomposer, self.sf)

    def forward(self, input_series):
        pending_data = None
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
    def __init__(self, args):
        super(HPMRESN, self).__init__()
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.num_decomposer = args.num_decomposer
        self.num_encoder = self.num_decomposer + 1
        self.smoothing_factor = args.sf
        self.leaking_rate_mean = args.lr_mean
        self.leaking_rate_std = args.lr_std
        self.in_scale_mean = args.in_scale_mean
        self.in_scale_std = args.in_scale_std
        self.res_density_mean = args.res_density_mean
        self.res_density_std = args.res_density_std
        self.regularization = args.C
        self.param_dist = args.param_dist
        self.decomposer = Decomposer(num_decomposer=args.num_decomposer, sf=args.sf)
        self.decoder = Decoder(num_encoder=self.num_encoder, regularization_factor=self.regularization)

    def generate_encoder(self):
        self.MR = Encoder(in_channels=args.in_channels, out_channels=args.out_channels, leaking_rate_mean=args.lr_mean,
                          leaking_rate_std=args.lr_std, in_scale_mean=args.in_scale_mean,
                          in_scale_std=args.in_scale_std, res_density_mean=args.res_density_mean,
                          res_density_std=args.res_density_std, num_encoder=self.num_encoder,
                          param_dist=self.param_dist)

    def extra_repr(self):
        return 'decomposer={}, encoder={}'.format(
            self.decomposer, self.encoder)

    def forward(self):
        pass


if __name__ == "__main__":
    model = HPMRESN(args)
    dataset = timeseries_loader(dataset_name=args.dataset, validation=args.validation, delay=args.delay)
    components = model.decomposer(dataset.loading_data)
    components_X = dict()
    components_y = dict()
    if dataset.validation == True:
        NRMSE_validation = []
    NRMSE_test = []
    for r in tqdm(range(args.num_trials)):
        model.generate_encoder()
        if dataset.validation == True:
            partial_validation_outputs = dict()
            validation_outputs = 0
        partial_test_outputs = dict()
        test_outputs = 0
        for d in range(args.num_decomposer):
            components_X['{}'.format(d)], components_y['{}'.format(d)] = input_target_spliter(
                components['trend_{}'.format(d)], dataset.delay)
            components_X['{}'.format(d)] = torch.Tensor(components_X['{}'.format(d)])
            components_X['{}'.format(d)] = components_X['{}'.format(d)].unsqueeze(1)
            temp_state = state_transform(getattr(model.MR, "encoder_{}".format(d))(components_X['{}'.format(d)]))
            getattr(model.decoder, "decoder_{}".format(d)).train(temp_state[dataset.train_set_index],
                                                                 components_y['{}'.format(d)][dataset.train_set_index])
            if dataset.validation == True:
                validation_outputs += getattr(model.decoder, "decoder_{}".format(d)).test(
                    temp_state[dataset.validation_set_index])
            test_outputs += getattr(model.decoder, "decoder_{}".format(d)).test(temp_state[dataset.test_set_index])

            if d == args.num_decomposer - 1:
                components_X['{}'.format(d + 1)], components_y['{}'.format(d + 1)] = input_target_spliter(
                    components['cycle_{}'.format(d)], dataset.delay)
                components_X['{}'.format(d + 1)] = torch.Tensor(components_X['{}'.format(d + 1)])
                components_X['{}'.format(d + 1)] = components_X['{}'.format(d + 1)].unsqueeze(1)
                temp_state = state_transform(
                    getattr(model.MR, "encoder_{}".format(d + 1))(components_X['{}'.format(d + 1)]))
                getattr(model.decoder, "decoder_{}".format(d + 1)).train(temp_state[dataset.train_set_index],
                                                                         components_y['{}'.format(d + 1)][
                                                                             dataset.train_set_index])
                if dataset.validation == True:
                    validation_outputs += getattr(model.decoder, "decoder_{}".format(d + 1)).test(
                        temp_state[dataset.validation_set_index])
                test_outputs += getattr(model.decoder, "decoder_{}".format(d + 1)).test(
                    temp_state[dataset.test_set_index])
        if dataset.validation == True:
            NRMSE_validation.append(
                mean_squared_error(y_true=dataset.target_y[dataset.validation_set_index], y_pred=validation_outputs,
                                   squared=False) / np.std(dataset.target_y[dataset.validation_set_index]))
        NRMSE_test.append(mean_squared_error(y_true=dataset.target_y[dataset.test_set_index], y_pred=test_outputs,
                                             squared=False) / np.std(dataset.target_y[dataset.test_set_index]))

        # train data plot
        plt.plot(dataset.target_y[dataset.train_set_index], label='True')
        plt.plot(dataset.target_y[dataset.train_set_index], label='Predicted')
        plt.legend()
        plt.show()

        # 予測画像の出力
        if args.output:
            plt.plot(dataset.target_y[dataset.test_set_index], label='True')
            plt.plot(test_outputs, label='Predicted')
            plt.legend()
            plt.show()
    if dataset.validation == True:
        print('mean RMSE and (std) on the validation set: {}({}) for {}'.format(np.mean(np.array(NRMSE_validation)),
                                                                                np.std(np.array(NRMSE_validation)),
                                                                                args.dataset))
    print('mean RMSE and (std) on the test set: {}({}) for {}'.format(np.mean(np.array(NRMSE_test)),
                                                                      np.std(np.array(NRMSE_test)), args.dataset))
