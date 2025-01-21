import logging
import os
import cv2
import random
import numpy as np
import torch
import wandb
import hydra
import multiprocessing as mp

from numba.cuda import current_context

from .base_sim import BaseSim
from .pusht_env import PushTEnv
from .utils import assign_process_to_cpu
import einops
from tqdm import tqdm
from torch.utils.data import DataLoader

from fast_mail.agents.utils.push_t_debug_utils import *

log = logging.getLogger(__name__)


class PushTSim(BaseSim):
    def __init__(
            self,
            num_episode,
            max_step_per_episode,
            seed,
            device,
            render,
            n_cores,
            n_test_tasks,
            debug_dataset,
            use_multiprocessing=False
    ):
        """
        num_episode: number of episodes to run for each task
        max_step_per_episode: ...


        """
        super().__init__(seed, device, render, n_cores)
        self.env_name = 'PushTSim'

        self.render = render
        self.num_episode = num_episode
        self.max_step_per_episode = max_step_per_episode

        self.success_rate = 0
        self.use_multiprocessing = use_multiprocessing
        self.n_tasks = n_test_tasks
        if debug_dataset is not None:
            self.debug_dataloader = DataLoader(debug_dataset, batch_size=1, shuffle=False, num_workers=0)

    def eval_agent(self,
                   agent,
                   #    model_state_dict,
                   #    scaler,
                   contexts,
                   success,
                   episode_lengths,
                   pid,
                   cpu_set,
                   counter,
                   agent_config=None,
                   ):
        """
        success: [num_tasks, num_episode]
        contexts: [num_tasks * num_episode]

        """

        # Only set CPU affinity if using multiprocessing
        if self.use_multiprocessing:
            print(os.getpid(), cpu_set)
            assign_process_to_cpu(os.getpid(), cpu_set)

        # agent = hydra.utils.instantiate(agent_config)
        # agent.recover_state(model_state_dict, scaler)

        print(contexts)
        env_args = {
            "legacy": False,
            "block_cog": None,
            "damping": None,
            "render_action": False,
            "render_size": 224,
            "reset_to_state": None,  # position x 2, position x2, angle
        }

        env = PushTEnv(**env_args)
        env.seed(self.seed)
        curr_context = contexts[0]
        count = -1
        for context in contexts:
            agent.reset()
            if curr_context == context:
                obs = env.reset()
                count += 1
            else:
                obs = env.reset_new_test_case()
                count = 0
            curr_context = context

            # multiprocessing simulation
            for j in range(self.max_step_per_episode):
                obs = einops.rearrange(obs, "H W C -> 1 1 C H W") / 255.0  # 1 view

                # Convert to torch tensor
                obs = torch.from_numpy(obs).float().to(self.device)

                obs_dict = {"agentview_rgb": obs,
                            "lang_emb": False}

                action = agent.predict(obs_dict)
                action = action.detach().cpu().numpy()
                obs, r, done, info = env.step(action)
                if self.render:
                    env.render(mode='human')

                if done:
                    success[curr_context,  count] = 1
                    episode_lengths[curr_context, count] = j + 1
                    break

            if episode_lengths[curr_context, count] == 0:
                episode_lengths[curr_context, count] = self.max_step_per_episode

            # env.close()
            if hasattr(counter, 'get_lock'):  # If it's a multiprocessing Value
                with counter.get_lock():
                    counter.value += 1
                    current_count = counter.value
            else:  # If it's a simple object with value attribute (single process)
                counter.value += 1
                current_count = counter.value
                counter.update()

            mask = episode_lengths.flatten() != 0
            completed_success = success.flatten()[mask]
            completed_lengths = episode_lengths.flatten()[mask]
            average_success = torch.mean(completed_success).item()
            average_episode_length = torch.mean(completed_lengths).item()
            print(f'completed_success {completed_success}')
            print(f'completed_lengths {completed_lengths}')
            print(f'average success rate: {average_success}')
            print(f'average episode length: {average_episode_length}')

    def debug_agent(self,
                   agent,
                   contexts,
                   success,
                   episode_lengths,
                   pid,
                   cpu_set,
                   counter,
                   agent_config=None,
                   ):
        # Only set CPU affinity if using multiprocessing
        if self.use_multiprocessing:
            print(os.getpid(), cpu_set)
            assign_process_to_cpu(os.getpid(), cpu_set)

        env = None
        # Get the first data from the dataloader
        cnt = 0
        final_sim_obs = None
        for data in self.debug_dataloader:
            obs, action, mask = data
            init_state = obs["state"][0, 0, :].detach().cpu().numpy()

            if env is None:
                env_args = {
                    # "bddl_file_name": task_bddl_file,
                    # "camera_heights": 128,
                    # "camera_widths": 128

                    "legacy": False,
                    "block_cog": None,
                    "damping": None,
                    "render_action": False,
                    "render_size": 224,
                    "reset_to_state": None,  # position x 2, position x2, angle
                }

                env = PushTEnv(**env_args)
                env.seed(self.seed)
                env.reset()
            else:
                env.reset_to_state = init_state
                env.reset()

            T = action.shape[1]
            for camera in obs.keys():
                if camera == 'lang':
                    continue

                obs[camera] = obs[camera].to(self.device)

                if 'rgb' not in camera and 'image' not in camera:
                    continue
                obs[camera] = obs[camera][:, :1].contiguous()
            pa = agent.predict(obs)

            for j in range(T):
                a = action[0, j, :].detach().cpu().numpy()
                pa = pa.detach().cpu().numpy()
                new_obs, r, done, info = env.step(a)

                new_obs = einops.rearrange(new_obs, "H W C -> 1 1 C H W") / 255.0  # 1 view

                # Convert to torch tensor
                new_obs = torch.from_numpy(new_obs).float().to(self.device)

                obs_dict = {"agentview_rgb": new_obs,
                            "lang_emb": False}
                pa = agent.predict(obs_dict)

            #     if self.render:
            #         env.render(mode='human')
            #
            #
            # plot_batch_images(new_obs, 1, "BT-CHW")
            print("done", action.shape)


            if hasattr(counter, 'get_lock'):  # If it's a multiprocessing Value
                with counter.get_lock():
                    counter.value += 1
                    current_count = counter.value
            else:  # If it's a simple object with value attribute (single process)
                counter.value += 1
                current_count = counter.value
                counter.update()

            # Roll out the agent
            if cnt == 1:
                final_sim_obs = new_obs

            if cnt == 9:
                compare_dataset_sim_observation(obs["agentview_rgb"], final_sim_obs)
                cnt = 1
            cnt += 1

    def get_task_embs(self, task_embs):
        self.task_embs = task_embs


    def test_agent(self, agent, cpu_set):
        if cpu_set is None:
            num_cpu = self.n_cores
            cpu_set = [i for i in range(num_cpu)]
        else:
            num_cpu = len(cpu_set)

        if self.use_multiprocessing:
            print("there is {} cpus".format(num_cpu))
        else:
            print("not using multiprocessing, run on 1 cpu")


        num_tasks = self.n_tasks
        success = torch.zeros([num_tasks, self.num_episode]).share_memory_()
        episode_lengths = torch.zeros([num_tasks, self.num_episode]).share_memory_()
        all_runs = num_tasks * self.num_episode

        contexts = np.arange(num_tasks)
        contexts = np.repeat(contexts, self.num_episode)

        print(contexts.shape, success.shape)

        context_ind = np.arange(self.num_episode)
        context_ind = np.tile(context_ind, num_tasks)

        if not self.use_multiprocessing:
            # Single process execution
            pbar = tqdm(total=all_runs, desc="Testing agent")
            counter = type('Counter', (), {'value': 0})()  # Simple counter object

            def update_pbar():
                pbar.update(1)

            counter.update = update_pbar  # Add update method to counter

            self.eval_agent(
                contexts=contexts,
                success=success,
                episode_lengths=episode_lengths,
                pid=0,
                cpu_set=set(cpu_set),
                counter=counter,
                agent=agent
            )
            pbar.close()