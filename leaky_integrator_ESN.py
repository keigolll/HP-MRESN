import argparse
import numpy as np
import torch
from torch.nn.modules import Module
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.utils import deprecated
from dataset_loader import timeseries_loader, input_target_spliter
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

# 引数パーサーの設定
parser = argparse.ArgumentParser()
parser.add_argument('--in_channels', type=int, default=1, help='Dimension of inputs')
parser.add_argument('--out_channels', type=int, default=5000, help='Dimension of embeddings')
parser.add_argument('--lr', type=float, default=0.4, help='Leaking rate, (0,1], default = 1.0')
parser.add_argument('--in_scale', type=float, default=1, help='Input scaling, default = 1.0')
parser.add_argument('--res_density', type=float, default=0.1, help='Density of each reservoir, (0,1]')
parser.add_argument('--C', type=float, default=1e-6, help='Regularization factor of the ridge regression')
parser.add_argument('--dataset', type=str, default='Sin_waves', help='Datasets used in the paper')
parser.add_argument('--validation', type=bool, default=True, help='Validation mode or not')
parser.add_argument('--num_trials', type=int, default=1, help='Number of random trials')
parser.add_argument('--delay', type=int, default=1, help='K-step-ahead, you can choose the value of K.')
# 予測画像を出力するか
parser.add_argument('--output', type=bool, default=True, help='Output prediction images')
args = parser.parse_args()


# Root Mean Squared Error function
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# モジュール定義
class LeakyIntegratorESN(Module):
    def __init__(self, input_dim, reservoir_dim, spectral_radius=0.95, scale_in=1.0, leaking_rate=0.3, density=0.1,
                 Win_assign='Uniform', W_assign='Uniform'):
        super(LeakyIntegratorESN, self).__init__()
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.spectral_radius = spectral_radius
        self.scale_in = scale_in
        self.leaking_rate = leaking_rate
        self.density = density
        self.Win_assign = Win_assign
        self.W_assign = W_assign
        self.Win = None
        self.W = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.Win_assign == 'Uniform':
            Win_np = np.random.uniform(-self.scale_in, self.scale_in, size=(self.input_dim + 1, self.reservoir_dim))
        elif self.Win_assign == 'Xavier':
            Win_np = (np.random.randn(self.input_dim + 1, self.reservoir_dim) / np.sqrt(
                self.input_dim + 1)) * self.scale_in
        elif self.Win_assign == 'Gaussian':
            Win_np = np.random.randn(self.input_dim + 1, self.reservoir_dim) * self.scale_in

        if self.W_assign == 'Uniform':
            if self.density < 1:
                W_np = np.zeros((self.reservoir_dim, self.reservoir_dim))
                for row in range(self.reservoir_dim):
                    number_row_elements = round(self.density * self.reservoir_dim)
                    row_elements = random.sample(range(self.reservoir_dim), int(number_row_elements))
                    W_np[row, row_elements] = np.random.uniform(-1, +1, size=(1, int(number_row_elements)))
            else:
                W_np = np.random.uniform(-1, +1, size=(self.reservoir_dim, self.reservoir_dim))

        elif self.W_assign == 'Gaussian':
            W_np = np.random.randn(self.reservoir_dim, self.reservoir_dim) if self.density >= 1 else np.zeros(
                (self.reservoir_dim, self.reservoir_dim))
            if self.density < 1:
                for row in range(self.reservoir_dim):
                    number_row_elements = round(self.density * self.reservoir_dim)
                    row_elements = random.sample(range(self.reservoir_dim), int(number_row_elements))
                    W_np[row, row_elements] = np.random.randn(1, int(number_row_elements))

        elif self.W_assign == 'Xavier':
            W_np = np.random.randn(self.reservoir_dim, self.reservoir_dim) / np.sqrt(
                self.reservoir_dim) if self.density >= 1 else np.zeros((self.reservoir_dim, self.reservoir_dim))
            if self.density < 1:
                for row in range(self.reservoir_dim):
                    number_row_elements = round(self.density * self.reservoir_dim)
                    row_elements = random.sample(range(self.reservoir_dim), int(number_row_elements))
                    W_np[row, row_elements] = np.random.randn(1, int(number_row_elements)) / np.sqrt(self.reservoir_dim)

        eig_values = np.linalg.eigvals(W_np)
        actual_sr = np.max(np.absolute(eig_values))
        W_np = (W_np * self.spectral_radius) / actual_sr

        self.Win = torch.Tensor(Win_np)
        self.W = torch.Tensor(W_np)

    def forward(self, x):
        h = torch.zeros(1, self.reservoir_dim)
        input_bias = torch.ones(x.shape[0], 1)

        if x.dim() == 3:
            input_bias = input_bias.unsqueeze(1)  # Adjust dimensions to match x if x is 3D

        x = torch.cat((x, input_bias), dim=-1)
        states = []

        for i in range(x.shape[0]):
            u = x[i, :].unsqueeze(0)
            if u.dim() == 3:
                u = u.squeeze(0)
            h = (1 - self.leaking_rate) * h + self.leaking_rate * torch.tanh(
                torch.mm(u, self.Win) + torch.mm(h, self.W))
            states.append(h)

        return torch.stack(states)


# メイン実行部分
if __name__ == "__main__":
    # データセットの読み込み
    dataset = timeseries_loader(dataset_name=args.dataset, validation=args.validation, delay=args.delay)

    # モデルのインスタンス化
    model = LeakyIntegratorESN(input_dim=args.in_channels, reservoir_dim=args.out_channels, spectral_radius=0.95,
                               scale_in=args.in_scale, leaking_rate=args.lr, density=args.res_density)

    # データの準備
    x_train_tensor = torch.Tensor(dataset.loading_data[dataset.train_set_index]).unsqueeze(1)
    y_train_tensor = torch.Tensor(dataset.target_y[dataset.train_set_index]).unsqueeze(1)
    x_val_tensor = torch.Tensor(dataset.loading_data[dataset.validation_set_index]).unsqueeze(1)
    y_val_tensor = torch.Tensor(dataset.target_y[dataset.validation_set_index]).unsqueeze(1)
    x_test_tensor = torch.Tensor(dataset.loading_data[dataset.test_set_index]).unsqueeze(1)
    y_test_tensor = torch.Tensor(dataset.target_y[dataset.test_set_index]).unsqueeze(1)

    # 次元を2次元に変換
    y_train_tensor = y_train_tensor.squeeze()
    y_val_tensor = y_val_tensor.squeeze()
    y_test_tensor = y_test_tensor.squeeze()

    # リッジ回帰モデルの定義
    regressor = Ridge(alpha=args.C)

    NRMSE_validation = []
    NRMSE_test = []

    for r in tqdm(range(args.num_trials)):
        # 訓練データに対する状態の取得
        train_states = model(x_train_tensor)
        train_states = train_states.squeeze().detach().numpy()

        # リッジ回帰の訓練
        regressor.fit(train_states, y_train_tensor)

        # 検証データに対する予測
        val_states = model(x_val_tensor)
        val_states = val_states.squeeze().detach().numpy()
        y_val_pred = regressor.predict(val_states)

        # テストデータに対する予測
        test_states = model(x_test_tensor)
        test_states = test_states.squeeze().detach().numpy()
        y_test_pred = regressor.predict(test_states)

        # 予測画像の出力
        if args.output:
            plt.plot(y_test_tensor, label='True')
            plt.plot(y_test_pred, label='Predicted')
            plt.legend()
            plt.show()



        # 評価
        if args.validation:
            val_rmse = root_mean_squared_error(y_val_tensor, y_val_pred) / torch.std(y_val_tensor.clone().detach())
            NRMSE_validation.append(val_rmse)
        test_rmse = root_mean_squared_error(y_test_tensor, y_test_pred) / torch.std(y_test_tensor.clone().detach())
        NRMSE_test.append(test_rmse)

    if args.validation:
        print(
            f'mean RMSE and (std) on the validation set: {np.mean(np.array(NRMSE_validation))}({np.std(np.array(NRMSE_validation))}) for {args.dataset}')
    print(
        f'mean RMSE and (std) on the test set: {np.mean(np.array(NRMSE_test))}({np.std(np.array(NRMSE_test))}) for {args.dataset}')
