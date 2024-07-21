import matplotlib.pyplot as plt
import os
import numpy as np
from diverse_grouped_ESN import evaluate_model
import threading

def plot_rmse_vs_std(parameter, fixed_mean, std_range, num_points, in_channels=1, out_channels=500, num_decomposer=10, regularization=1e-6, dataset_name='Bike', validation=True, num_trials=20, delay=1,output=False):
    std_values = np.linspace(std_range[0], std_range[1], num_points)
    mean_rmse_values = []
    std_rmse_values = []
    i = 0

    for std in std_values:
        i += 1
        print(f'Iteration {i}/{num_points}')
        if parameter == 'leaking_rate':
            mean_val_rmse, std_val_rmse, mean_test_rmse, std_test_rmse = evaluate_model(
                in_channels=in_channels, out_channels=out_channels, num_esn=num_decomposer,
                lr_mean=fixed_mean, lr_std=std, in_scale_mean=1.0, in_scale_std=0.1, res_density_mean=0.1, res_density_std=0.02,
                regularization=regularization, dataset_name=dataset_name, validation=validation, num_trials=num_trials, delay=delay ,output=output
            )
        elif parameter == 'in_scale':
            mean_val_rmse, std_val_rmse, mean_test_rmse, std_test_rmse = evaluate_model(
                in_channels=in_channels, out_channels=out_channels, num_esn=num_decomposer,
                lr_mean=0.4, lr_std=0.1, in_scale_mean=fixed_mean, in_scale_std=std, res_density_mean=0.1, res_density_std=0.02,
                regularization=regularization,  dataset_name=dataset_name, validation=validation, num_trials=num_trials, delay=delay ,output=output
            )
        elif parameter == 'res_density':
            mean_val_rmse, std_val_rmse, mean_test_rmse, std_test_rmse = evaluate_model(
                in_channels=in_channels, out_channels=out_channels, num_esn=num_decomposer,
                lr_mean=0.4, lr_std=0.1, in_scale_mean=1.0, in_scale_std=0.1, res_density_mean=fixed_mean, res_density_std=std,
                regularization=regularization,  dataset_name=dataset_name, validation=validation, num_trials=num_trials, delay=delay ,output=output
            )
        else:
            raise ValueError("Unsupported parameter type")

        mean_rmse_values.append(mean_test_rmse)
        std_rmse_values.append(std_test_rmse)

    plt.figure(figsize=(10, 6))
    plt.errorbar(std_values, mean_rmse_values, yerr=std_rmse_values, fmt='-o', capsize=5, label='Mean RMSE')
    plt.xlabel(f'{parameter} standard deviation')
    plt.ylabel('Mean RMSE')
    plt.title(f'Mean RMSE vs. {parameter} standard deviation\n(Fixed mean: {fixed_mean})')
    plt.legend()
    plt.grid(True)

    # 保存用ディレクトリを作成
    os.makedirs('fig', exist_ok=True)
    plt.savefig(f'fig/rmse_vs_std_{parameter}_mean_{fixed_mean}_std_{std_range[0]}-{std_range[1]}_points_{num_points}.png')
    plt.close()

def plot_rmse_vs_std_thread(parameter, fixed_mean, std_range, num_points, in_channels=1, out_channels=500, num_decomposer=10, regularization=1e-6, dataset_name='Bike', validation=True, num_trials=20, delay=1,output=False):
    std_values = np.linspace(std_range[0], std_range[1], num_points)

    th = threading.Thread(target=plot_rmse_vs_std, args=(parameter, fixed_mean, std_range, num_points, in_channels, out_channels, num_decomposer, regularization, dataset_name, validation, num_trials, delay,output))

    th.start()



if __name__ == '__main__':



    plot_rmse_vs_std(
        parameter='leaking_rate',
        fixed_mean=0.5,
        std_range=(0, 0.5),
        num_points=50,
        in_channels=1,
        out_channels=500,
        num_decomposer=10,
        regularization=1e-6,
        dataset_name='Sin_waves',
        validation=True,
        num_trials=20,
        delay=1
    )

    plot_rmse_vs_std(
        parameter='in_scale',
        fixed_mean=0.8,
        std_range=(0, 0.2),
        num_points=50,
        in_channels=1,
        out_channels=500,
        regularization=1e-6,
        dataset_name='Sin_waves',
        validation=True,
        num_trials=20,
        delay=1
    )

    plot_rmse_vs_std(
        parameter='res_density',
        fixed_mean=0.1,
        std_range=(0, 0.1),
        num_points=50,
        in_channels=1,
        out_channels=500,
        num_decomposer=10,
        regularization=1e-6,
        dataset_name='Sin_waves',
        validation=True,
        num_trials=20,
        delay=1
    )
