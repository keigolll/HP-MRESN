import argparse
import numpy as np
from torch.nn.modules import Module
import ESN
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from dataset_loader import timeseries_loader, input_target_spliter
from tqdm import tqdm
import matplotlib.pyplot as plt


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


class SimpleESN(Module):
    def __init__(self, input_dim, reservoir_dim, spectral_radius=0.95, scale_in=1.0, leaking_rate=0.3, density=0.1):
        super(SimpleESN, self).__init__()
        self.esn = ESN.ESN(input=input_dim, reservoir=reservoir_dim, sr=spectral_radius, scale_in=scale_in,
                           leaking_rate=leaking_rate, density=density)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        states = self.esn(x)
        return torch.stack(states)


class GroupedESN(Module):
    def __init__(self, num_esn, input_dim, reservoir_dim, lr_mean, lr_std, in_scale_mean, in_scale_std,
                 res_density_mean, res_density_std, spectral_radius=0.95):
        super(GroupedESN, self).__init__()
        self.esns = []
        self.num_esn = num_esn
        for _ in range(num_esn):
            lr = np.clip(np.random.normal(lr_mean, lr_std), 0.0001, 1)
            in_scale = np.clip(np.random.normal(in_scale_mean, in_scale_std), 0.1, 10)
            res_density = np.clip(np.random.normal(res_density_mean, res_density_std), 0.0001, 1)
            self.esns.append(SimpleESN(input_dim, reservoir_dim, spectral_radius, in_scale, lr, res_density))

    def forward(self, x):
        all_states = []
        for esn in self.esns:
            state = esn(x)
            all_states.append(state[:, -1, :].unsqueeze(0))
        return torch.cat(all_states, dim=2).squeeze(0)


def evaluate_model(in_channels, out_channels, num_esn, lr_mean, lr_std, in_scale_mean, in_scale_std,
                   res_density_mean, res_density_std, regularization, dataset_name, validation, num_trials, delay,
                   output):
    dataset = timeseries_loader(dataset_name=dataset_name, validation=validation, delay=delay)

    model = GroupedESN(num_esn=num_esn, input_dim=in_channels, reservoir_dim=out_channels, lr_mean=lr_mean,
                       lr_std=lr_std, in_scale_mean=in_scale_mean, in_scale_std=in_scale_std,
                       res_density_mean=res_density_mean, res_density_std=res_density_std)

    x_train_tensor = torch.Tensor(dataset.loading_data[dataset.train_set_index]).unsqueeze(1)
    y_train_tensor = torch.Tensor(dataset.target_y[dataset.train_set_index]).unsqueeze(1)
    x_val_tensor = torch.Tensor(dataset.loading_data[dataset.validation_set_index]).unsqueeze(1)
    y_val_tensor = torch.Tensor(dataset.target_y[dataset.validation_set_index]).unsqueeze(1)
    x_test_tensor = torch.Tensor(dataset.loading_data[dataset.test_set_index]).unsqueeze(1)
    y_test_tensor = torch.Tensor(dataset.target_y[dataset.test_set_index]).unsqueeze(1)

    y_train_tensor = y_train_tensor.squeeze()
    y_val_tensor = y_val_tensor.squeeze()
    y_test_tensor = y_test_tensor.squeeze()

    regressor = Ridge(alpha=regularization)

    NRMSE_validation = []
    NRMSE_test = []

    for r in tqdm(range(num_trials)):
        train_states = model(x_train_tensor)
        train_states = train_states.squeeze().detach().numpy()

        regressor.fit(train_states, y_train_tensor)

        val_states = model(x_val_tensor)
        val_states = val_states.squeeze().detach().numpy()
        y_val_pred = regressor.predict(val_states)

        test_states = model(x_test_tensor)
        test_states = test_states.squeeze().detach().numpy()
        y_test_pred = regressor.predict(test_states)

        if validation:
            val_rmse = root_mean_squared_error(y_val_tensor, y_val_pred) / torch.std(y_val_tensor.clone().detach())
            NRMSE_validation.append(val_rmse)
        test_rmse = root_mean_squared_error(y_test_tensor, y_test_pred) / torch.std(y_test_tensor.clone().detach())
        NRMSE_test.append(test_rmse)

        if output:
            plt.plot(y_test_tensor, label='True')
            plt.plot(y_test_pred, label='Predicted')
            plt.legend()
            plt.show()

    if validation:
        print(
            f'mean RMSE and (std) on the validation set: {np.mean(np.array(NRMSE_validation))}({np.std(np.array(NRMSE_validation))}) for {dataset_name}')
    print(
        f'mean RMSE and (std) on the test set: {np.mean(np.array(NRMSE_test))}({np.std(np.array(NRMSE_test))}) for {dataset_name}')

    mean_validation_rmse = np.mean(np.array(NRMSE_validation)) if validation else None
    std_validation_rmse = np.std(np.array(NRMSE_validation)) if validation else None
    mean_test_rmse = np.mean(np.array(NRMSE_test))
    std_test_rmse = np.std(np.array(NRMSE_test))

    return mean_validation_rmse, std_validation_rmse, mean_test_rmse, std_test_rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_channels', type=int, default=1, help='Dimension of inputs')
    parser.add_argument('--out_channels', type=int, default=500, help='Dimension of embeddings')
    parser.add_argument('--num_esn', type=int, default=5, help='Number of ESNs in the group')
    parser.add_argument('--lr_mean', type=float, default=0.5, help='Mean of leaking rate, (0,1]')
    parser.add_argument('--lr_std', type=float, default=0.5, help='Standard deviation of leaking rate')
    parser.add_argument('--in_scale_mean', type=float, default=1.0, help='Mean of input scaling')
    parser.add_argument('--in_scale_std', type=float, default=0, help='Standard deviation of input scaling')
    parser.add_argument('--res_density_mean', type=float, default=0.1, help='Mean of reservoir density, (0,1]')
    parser.add_argument('--res_density_std', type=float, default=0, help='Standard deviation of reservoir density')
    parser.add_argument('--C', type=float, default=1e-6, help='Regularization factor of the ridge regression')
    parser.add_argument('--dataset', type=str, default='Sin_waves', help='Datasets used in the paper')
    parser.add_argument('--validation', type=bool, default=True, help='Validation mode or not')
    parser.add_argument('--num_trials', type=int, default=1, help='Number of random trials')
    parser.add_argument('--delay', type=int, default=1, help='K-step-ahead, you can choose the value of K.')
    parser.add_argument('--output', type=bool, default=True, help='Output prediction images')
    args = parser.parse_args()

    evaluate_model(args.in_channels, args.out_channels, args.num_esn, args.lr_mean, args.lr_std, args.in_scale_mean,
                   args.in_scale_std,
                   args.res_density_mean, args.res_density_std, args.C, args.dataset, args.validation, args.num_trials,
                   args.delay, args.output)
