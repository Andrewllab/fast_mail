import wandb2numpy

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['text.usetex'] = True


def plot_heat_map(mean, variance=None, plot_variance=False, title='Heatmap'):

    mean = mean.astype(float)
    if variance is not None:
        mean_label = mean#.to_numpy()
        mean_label = np.around(mean_label, decimals=3)
        # mean_label = mean_label.astype(str)

        variance = variance.astype(float)
        variance = variance#.to_numpy()
        variance = np.around(variance, decimals=3)
        # variance = variance.astype(str)
        grid_string = np.zeros(shape=(variance.shape[0], variance.shape[1]), dtype=object)

        print(variance)

        for i in range(variance.shape[0]):
            for j in range(variance.shape[1]):
                # variance[i, j] = f'${mean_label[i, j]}$\n' + r"$\scriptstyle{{\pm}" + variance[i,j] +r"}$"
                grid_string[i, j]= f'{mean_label[i, j]:.2f}\n' + r"$\scriptstyle{{\pm}" + f'{variance[i,j]:.2f}' +r"}$"

    sns.set('paper', 'white',
            rc={'font.size': 16, 'axes.labelsize': 18,
                'legend.fontsize': 20, 'axes.titlesize': 20,
                'xtick.labelsize': 16, 'ytick.labelsize': 16,
                "pgf.rcfonts": False,
                })
    plt.rc('font', **{'family': 'serif', 'serif': ['Times']})
    plt.rc('text', usetex=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    colormap = sns.color_palette("coolwarm", as_cmap=True)
    # colormap = sns.color_palette("rocket", as_cmap=True)
    # colormap = sns.color_palette("magma", as_cmap=True)
    # colormap = sns.color_palette("rocket_r", as_cmap=True)

    column_names = ['2', '4', '6', '8']
    index = ['8', '6', '4', '2']

    df = pd.DataFrame(mean, index=index, columns=column_names)

    if plot_variance:
        ax = sns.heatmap(df, annot=grid_string, fmt='', linewidths=.5, cbar=False, cmap=colormap)
    else:
        ax = sns.heatmap(mean, linewidths=.5, cbar=False, cmap=colormap)

    # ax.set(xlabel="----", ylabel="History")

    # plt.xlabel('-', color='White')  # Set the color of the xlabel to blue
    plt.ylabel('Decoder Blocks')
    plt.xlabel('Encoder Blocks')

    plt.title(title)
    plt.tight_layout()
    # tikzplotlib.save("ablation_sparse_success_rate.tex")
    plt.savefig(title + '.pdf')
    plt.show()


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


config = {
    "local": {
        'entity': 'tiger_or_cat',
        'project': 'david_robocasa_ablation',
        'groups': '',
        'fields': '',
        'runs': ["all"],
        'config': '',
        'output_path': '',
    }}


if __name__ == '__main__':

    # sv_dir = 'benchmark_results'
    # os.makedirs(sv_dir, exist_ok=True)

    # task_suite = ["libero_object", "libero_spatial", "libero_10"]

    store = ['max', 'mean']

    groups = ['beso_encoder_decoder']
    fields = ['TurnOnStove_average_success', 'CoffeeServeMug_average_success']

    # csv_file = open(sv_dir + '/bc_dec_results.csv', 'w', newline='')
    # writer = csv.writer(csv_file)

    for group in groups:

        if 'bc' in group:
            agent_names = ['bc_transformer', 'bc_mamba', 'bc_xlstm']
        elif 'beso' in group:
            agent_names = ['beso_transformer', 'beso_mamba', 'beso_xlstm']
        elif 'vqbet' in group:
            agent_names = ['vqbet_transformer', 'vqbet_mamba', 'vqbet_xlstm']
        elif 'fm' in group:
            agent_names = ['fm_transformer', 'fm_mamba', 'fm_xlstm']
        else:
            raise ValueError('Invalid group')

        agent_names = ['beso_xlstm']

        for agent_name in agent_names:

            # for task in task_suite:
            strings = []

            success_matrix = np.zeros((4, 4))
            success_matrix_std = np.zeros((4, 4))

            for i, num_encoders in enumerate([2, 4, 6, 8]):
                for j, num_decoders in enumerate([8, 6, 4, 2]):

                    tasks_success = []
                    tasks_std = []

                    # success_list = []
                    for evaluation in fields:

                        # writer.writerow([f'{task}'])
                        # writer.writerow(['mean', 'max'])

                        config['local']['groups'] = [group]
                        config['local']['fields'] = [evaluation]

                        config['local']['config'] = {
                            'scaler_type': {
                                'values': ['minmax']
                            },
                            'agent_name': {
                                'values': [agent_name]
                            },
                            'xlstm_encoder_blocks': {
                                'values': [num_encoders]
                            },
                            'xlstm_decoder_blocks': {
                                'values': [num_decoders]
                            }
                        }

                        data_dict, config_list = wandb2numpy.export_data(config)

                        if len(data_dict) == 0:
                            continue

                        success = data_dict['local'][evaluation]
                        success = np.squeeze(success)

                        mean_result = success.mean() * 100
                        std_result = success.std() * 100

                        tasks_success.append(mean_result)
                        tasks_std.append(std_result)

                        # mean_result = np.around(mean_result, decimals=1)
                        # std_result = np.around(std_result, decimals=1)

                        # number_string = f"& ${mean_result:.1f} \scriptstyle \pm {std_result:.1f}$"
                        # print(bcolors.WARNING + number_string + bcolors.ENDC)

                    success_matrix[j, i] = np.around(np.mean(tasks_success), decimals=1)
                    success_matrix_std[j, i] = np.around(np.mean(tasks_std), decimals=1)

            plot_heat_map(success_matrix, success_matrix_std, plot_variance=True, title=agent_name)


