import wandb2numpy
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from agents.utils.sim_path import sim_framework_path

# COLORS = ['tab:blue', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:grey',
#           'tab:olive', 'tab:cyan'] * 10
COLORS = ['tab:blue', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:grey',
          'tab:olive', 'tab:cyan'] * 10

COLOR_DICT = {
    'dis': 'tab:purple',
    'dis_ud': 'tab:purple',
    'FM': 'tab:red',
    'uha': 'tab:red',
    'mcd': 'tab:green',
    'ldvi': 'tab:green',
    'BESO': 'tab:blue',
    'cmcd_ud': 'tab:blue',
    'DDPM': 'tab:orange',
    'hbs': 'tab:orange',
}


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
        'project': 'david_robocasa_diffusion_steps',
        'groups': [],
        'fields': [],
        'runs': ["all"],
        'config': '',
        'output_path': '',
        # 'history_samples': 60
    }}

if __name__ == '__main__':
    steps = [1, 4, 8, 12, 16]
    # steps = [4, 8]
    fields = ['TurnOnStove_average_success']

    groups = ['beso_decoder_only_benchmark', 'fm_decoder_only_benchmark', 'ddpm_decoder_only_benchmark']

    fig, ax = plt.subplots()

    for group in groups:

        if 'bc' in group:
            agent_name = 'bc_xlstm' #['bc_transformer', 'bc_mamba', 'bc_xlstm']
            label = 'bc'
        elif 'beso' in group:
            agent_name = 'beso_xlstm' #['beso_transformer', 'beso_mamba', 'beso_xlstm']
            label = 'BESO'
        elif 'ddpm' in group:
            agent_name = 'ddpm_xlstm' #['ddpm_transformer', 'ddpm_mamba', 'ddpm_xlstm']
            label = 'DDPM'
        elif 'fm' in group:
            agent_name = 'fm_xlstm' #['fm_transformer', 'fm_mamba', 'fm_xlstm']
            label = 'FM'
        else:
            raise ValueError('Invalid group')

        # for task in task_suite:
        strings = []

        mean_results = []

        for step in steps:

            for evaluation in fields:

                config['local']['groups'] = [group]
                config['local']['fields'] = [evaluation]

                config['local']['config'] = {
                    'scaler_type': {
                        'values': ['minmax']
                    },
                    'agent_name': {
                        'values': [agent_name]
                    },
                    'num_sampling_steps': {
                        'values': [step]
                    },
                }

                data_dict, config_list = wandb2numpy.export_data(config)

                if len(data_dict) == 0:
                    continue

                success = data_dict['local'][evaluation]
                success = np.squeeze(success)

                mean_result = success.mean() * 100
                std_result = success.std() * 100

                number_string = f"& ${mean_result:.1f} \scriptstyle \pm {std_result:.1f}$"
                print(bcolors.WARNING + number_string + bcolors.ENDC)

                strings.append(number_string)

                mean_results.append(mean_result)

        ax.plot(np.arange(len(steps)), mean_results, 'x--', label=label,
                color=COLOR_DICT[label], )
        # ax.fill_between(np.arange(len(steps_done)), np.array(mean_results) - np.array(std_results),
        #                  np.array(mean_results) + np.array(std_results), color=COLOR_DICT[alg], alpha=0.3)

                # ax.scatter(np.arange(len(steps)), mean_results, color=COLORS[i // 2])
    plt.xticks(np.arange(len(steps)), steps)
    plt.grid()
    plt.ylabel('ESS [%]')
    plt.xlabel('$K$')
    tikzplotlib.save(sim_framework_path(f"utils/figures/diffusion_steps.tex"))
    plt.legend()
    plt.show()

