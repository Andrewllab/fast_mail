import csv
import os
import wandb2numpy
import matplotlib.pyplot as plt
import numpy as np


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
        'project': 'icml_libero_final',
        'groups': '',
        'fields': '',
        'runs': ["all"],
        'config': '',
        'output_path': '',
    }}

if __name__ == '__main__':

    # sv_dir = 'benchmark_results'
    # os.makedirs(sv_dir, exist_ok=True)

    # task_suite = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    task_suite = ["libero_spatial", "libero_10"]

    # groups = ['bc_decoder_only', 'bc_encoder_decoder', 'beso_decoder_only', 'beso_encoder_decoder', 'fm_decoder_only']
    groups = ['beso_encoder_decoder']
    fields = ['epoch100_average_success']

    # csv_file = open(sv_dir + '/bc_dec_results.csv', 'w', newline='')
    # writer = csv.writer(csv_file)

    for group in groups:

        if 'bc' in group:
            agent_names = ['bc_transformer', 'bc_mamba', 'bc_xlstm']
        elif 'beso' in group:
            # agent_names = ['beso_transformer', 'beso_mamba', 'beso_xlstm']
            agent_names = ['beso_mamba']
        elif 'vqbet' in group:
            agent_names = ['vqbet_transformer', 'vqbet_mamba', 'vqbet_xlstm']
        elif 'fm' in group:
            agent_names = ['fm_transformer', 'fm_mamba', 'fm_xlstm']
        else:
            raise ValueError('Invalid group')

        for agent_name in agent_names:

            strings = []
            task_success_rates = {10: [],
                                  50: []}

            for task in task_suite:

                # writer.writerow([f'{task}'])
                # writer.writerow(['mean', 'max'])

                # success_list = []
                for evaluation in fields:

                    for num_traj in [10, 50]:

                        for num_enc in [6, 8]:
                            for num_dec in [6, 8]:

                                config['local']['groups'] = [group]
                                config['local']['fields'] = [evaluation]

                                config['local']['config'] = {
                                    'task_suite': {
                                        'values': [task]
                                    },
                                    'agent_name': {
                                        'values': [agent_name]
                                    },
                                    'traj_per_task': {
                                        'values': [num_traj]
                                    },
                                    'scaler_type': {
                                        'values': ['minmax']
                                    },
                                    'mamba_n_layer_encoder': {
                                        'values': [num_enc]
                                    },
                                    'mamba_n_layer_decoder': {
                                        'values': [num_dec]
                                    },
                                }

                                data_dict, config_list = wandb2numpy.export_data(config)

                                if len(data_dict) == 0 or len(data_dict['local'][evaluation]) == 0:
                                    strings.append("& -")
                                    continue

                                success = data_dict['local'][evaluation]
                                success = np.squeeze(success)

                                mean_result = success.mean() * 100
                                std_result = success.std() * 100

                                print(f"Task: {task}, Traj: {num_traj}, Enc: {num_enc}, Dec: {num_dec}")

                                number_string = f"& ${mean_result:.1f} \scriptstyle \pm {std_result:.1f}$"
                                print(bcolors.WARNING + number_string + bcolors.ENDC)

                                strings.append(number_string)

                                # task_success_rates[num_traj].append(mean_result)
                        # task_success_rates.append(mean_result)
                    # success_list.append(success)

                # success_list = np.concatenate(success_list, axis=-1)
                # metric_means = success_list.mean(axis=-1)
                # metric_maxes = success_list.max(axis=-1)
                # writer.writerow([str(round(metric_means.mean(), 3)) + '+-' + str(round(metric_means.std(), 3)),
                #                  str(round(metric_maxes.mean(), 3)) + '+-' + str(round(metric_maxes.std(), 3))])
                #
                # writer.writerow([''])
                # writer.writerow([''])

            # traj10_task_success_rates = np.array(task_success_rates[10]).mean()
            # traj50_task_success_rates = np.array(task_success_rates[50]).mean()
            # # mean_string = f"& ${mean_task_success_rates:.1f}$"
            #
            # print(bcolors.WARNING + f'-------------------{group},{agent_name}-------------------\n' + bcolors.ENDC)
            # print(agent_names, task_suite)
            # for string in strings:
            #     print(string)
            # print(f"& ${traj10_task_success_rates:.1f}$")
            # print(f"& ${traj50_task_success_rates:.1f}$")
