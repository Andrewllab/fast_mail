import logging
import os
import cv2
import random
import numpy as np
import torch
import wandb
import hydra
import multiprocessing as mp
from .base_sim import BaseSim
from .pusht_env import PushTEnv
from .utils import assign_process_to_cpu
import einops
from tqdm import tqdm

log = logging.getLogger(__name__)


class PushTSim(BaseSim):
    def __init__(
            self,
            num_episode,
            max_step_per_episode,
            task_suite: str,
            use_eye_in_hand: bool,
            seed,
            device,
            render,
            n_cores,
            use_multiprocessing=False
    ):
        super().__init__(seed, device, render, n_cores)
        self.env_name = 'PushTSim'

        # according to the task_id, load the corresponding bddl file
        self.task_suite = task_suite

        self.use_eye_in_hand = use_eye_in_hand
        self.render = render

        self.num_episode = num_episode
        self.max_step_per_episode = max_step_per_episode

        self.success_rate = 0
        self.use_multiprocessing = use_multiprocessing

    def eval_agent(self,
                   agent,
                   #    model_state_dict,
                   #    scaler,
                   contexts,
                   context_ind,
                   success,
                   episode_lengths,
                   pid,
                   cpu_set,
                   counter):
        # Only set CPU affinity if using multiprocessing
        if self.use_multiprocessing:
            print(os.getpid(), cpu_set)
            assign_process_to_cpu(os.getpid(), cpu_set)

        # agent = hydra.utils.instantiate(agent_config)
        # agent.recover_state(model_state_dict, scaler)

        print(contexts)

        for i, context in enumerate(contexts):

            # task_suite = benchmark.get_benchmark_dict()[self.task_suite]()
            #
            # task_bddl_file = task_suite.get_task_bddl_file_path(context)
            #
            # file_name = os.path.basename(task_bddl_file).split('.')[0]
            #
            # task_emb = self.task_embs[file_name]

            # TODO Xi: Get initial states from the task suite

            # init_states = task_suite.get_task_init_states(context)

            env_args = {
                # "bddl_file_name": task_bddl_file,
                # "camera_heights": 128,
                # "camera_widths": 128

                "legacy": False,
                "block_cog": None,
                "damping": None,
                "render_action": True,
                "render_size": 224,
                "reset_to_state": None,
            }

            env = PushTEnv(**env_args)

            agent.reset()
            env.seed(self.seed)
            env.reset()
            # obs = env._set_state(init_states[context_ind[i]])

            # dummy actions all zeros for initial physics simulation
            dummy = np.zeros(7)
            dummy[-1] = -1.0  # set the last action to -1 to open the gripper

            # Observations: agent position(?), block position(?), block angle(?)
            for _ in range(5):
                obs, _, _, info = env.step(dummy)

            # multiprocessing simulation
            for j in range(self.max_step_per_episode):
                # TODO Xi: Update the observation dictionary
                # agentview_rgb = obs["agentview_image"]
                # eye_in_hand_rgb = obs["robot0_eye_in_hand_image"]
                #
                # joint_state = obs["robot0_joint_pos"]
                # gripper_state = obs["robot0_gripper_qpos"]
                #
                # robot_states = np.concatenate([joint_state, gripper_state], axis=-1)


                # obs_dict = {"agentview_rgb": agentview_rgb,
                #             "eye_in_hand_rgb": eye_in_hand_rgb,
                #             "lang_emb": task_emb,
                #             "robot_states": robot_states}

                # ToDo XI: Use this for the new observation dictionary
                return_obs = info["image"]
                return_obs = einops.rearrange(return_obs, "H W C -> 1 C H W") / 255.0  # 1 view

                obs_dict = {}

                action = agent.predict(obs_dict)
                obs, r, done, info = env.step(action)

                # if self.render:
                #     env.render()

                if r == 1:
                    success[context, context_ind[i]] = r
                    episode_lengths[context, context_ind[i]] = j + 1
                    break

            if success[context, context_ind[i]] == 0:
                episode_lengths[context, context_ind[i]] = self.max_step_per_episode

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

            env.close()

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

        if self.task_suite == "libero_90":
            num_tasks = 90
        else:
            num_tasks = 10

        success = torch.zeros([num_tasks, self.num_episode]).share_memory_()
        episode_lengths = torch.zeros([num_tasks, self.num_episode]).share_memory_()
        all_runs = num_tasks * self.num_episode

        contexts = np.arange(num_tasks)
        contexts = np.repeat(contexts, self.num_episode)

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
                context_ind=context_ind,
                success=success,
                episode_lengths=episode_lengths,
                pid=0,
                cpu_set=set(cpu_set),
                counter=counter,
                agent=agent
            )
            pbar.close()