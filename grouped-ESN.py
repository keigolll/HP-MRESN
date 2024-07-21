import argparse
import numpy as np
import torch
from torch.nn.modules import Module
import ESN
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from dataset_loader import timeseries_loader, input_target_spliter
from tqdm import tqdm
import matplotlib.pyplot as plt

# 引数パーサーの設定
parser = argparse.ArgumentParser()
parser.add_argument('--in_channels', type=int, default=1, help='Dimension of inputs')
parser.add_argument('--out_channels', type=int, default=500, help='Dimension of embeddings')
parser.add_argument('--num_esn', type=int, default=5, help='Number of ESNs in the group')
parser.add_argument('--lr', type=float, default=0.5, help='Leaking rate, (0,1], default = 1.0')
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

# モジュール定義

class SimpleESN(Module):
    def __init__(self, input_dim, reservoir_dim, spectral_radius=0.95, scale_in=1.0, leaking_rate=0.3, density=0.1):
        super(SimpleESN, self).__init__()
        self.esn = ESN.ESN(input=input_dim, reservoir=reservoir_dim, sr=spectral_radius, scale_in=scale_in, leaking_rate=leaking_rate, density=density)

    def forward(self, x):
        # 入力テンソルの次元を確認し、適切に変換
        if x.dim() == 3:
            x = x.squeeze(1)
        states = self.esn(x)
        return torch.stack(states)

class GroupedESN(Module):
    def __init__(self, num_esn, input_dim, reservoir_dim, spectral_radius=0.95, scale_in=1.0, leaking_rate=0.3, density=0.1):
        super(GroupedESN, self).__init__()
        self.esns = [SimpleESN(input_dim, reservoir_dim, spectral_radius, scale_in, leaking_rate, density) for _ in range(num_esn)]
        self.num_esn = num_esn

    def forward(self, x):
        all_states = []
        for esn in self.esns:
            state = esn(x)
            all_states.append(state[:, -1, :].unsqueeze(0))
        return torch.cat(all_states, dim=2).squeeze(0)

# メイン実行部分
if __name__ == "__main__":
    # データセットの読み込み
    dataset = timeseries_loader(dataset_name=args.dataset, validation=args.validation, delay=args.delay)

    # モデルのインスタンス化
    model = GroupedESN(num_esn=args.num_esn, input_dim=args.in_channels, reservoir_dim=args.out_channels, spectral_radius=0.95, scale_in=args.in_scale, leaking_rate=args.lr, density=args.res_density)

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

        # 評価
        if args.validation:
            val_rmse = mean_squared_error(y_val_tensor, y_val_pred, squared=False) / torch.std(y_val_tensor.clone().detach())
            NRMSE_validation.append(val_rmse)
        test_rmse = mean_squared_error(y_test_tensor, y_test_pred, squared=False) / torch.std(y_test_tensor.clone().detach())
        NRMSE_test.append(test_rmse)

        # 予測画像の出力
        if args.output:
            plt.plot(y_test_tensor, label='True')
            plt.plot(y_test_pred, label='Predicted')
            plt.legend()
            plt.show()



    if args.validation:
        print(f'mean RMSE and (std) on the validation set: {np.mean(np.array(NRMSE_validation))}({np.std(np.array(NRMSE_validation))}) for {args.dataset}')
    print(f'mean RMSE and (std) on the test set: {np.mean(np.array(NRMSE_test))}({np.std(np.array(NRMSE_test))}) for {args.dataset}')
