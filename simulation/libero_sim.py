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
# from libero.libero.envs import *
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

log = logging.getLogger(__name__)


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


def process_image_input(img_tensor):
    # return (img_tensor / 255. - 0.5) * 2.
    return img_tensor / 255.

class MultiTaskSim(BaseSim):
    def __init__(self,
                 num_episode,
                 max_step_per_episode,
                 task_suite: str,
                 use_eye_in_hand: bool,
                 seed,
                 device,
                 render,
                 n_cores):
        super().__init__(seed, device, render, n_cores)

        # according to the task_id, load the corresponding bddl file
        self.task_suite = task_suite

        self.use_eye_in_hand = use_eye_in_hand
        self.render = render

        self.num_episode = num_episode
        self.max_step_per_episode = max_step_per_episode

        self.success_rate = 0

    def reverse_rgb_channels(self, test_img):

        test_img = test_img[::-1, ::-1, :]
        # cv2.imshow("test_img", test_img)
        # cv2.waitKey(0)

        return np.ascontiguousarray(test_img)

    def eval_agent(self, agent_config, model_state_dict, scaler, contexts, context_ind, success, pid, cpu_set):
        print(os.getpid(), cpu_set)
        assign_process_to_cpu(os.getpid(), cpu_set)

        agent = hydra.utils.instantiate(agent_config)
        agent.recover_state(model_state_dict, scaler)

        print(contexts)

        for i, context in enumerate(contexts):

            task_suite = benchmark.get_benchmark_dict()[self.task_suite]()

            task_bddl_file = task_suite.get_task_bddl_file_path(context)

            file_name = os.path.basename(task_bddl_file).split('.')[0]

            task_emb = self.task_embs[file_name]

            # goal_images = self.goal_dicts[file_name]
            # goal_image = random.choice(goal_images)

            init_states = task_suite.get_task_init_states(context)

            env_args = {
                "bddl_file_name": task_bddl_file,
                "camera_heights": 128,
                "camera_widths": 128
            }

            env = OffScreenRenderEnv(**env_args)

            agent.reset()
            env.seed(self.seed)
            env.reset()
            obs = env.set_init_state(init_state=init_states[context_ind[i]])

            # dummy actions all zeros for initial physics simulation
            dummy = np.zeros(7)
            dummy[-1] = -1.0  # set the last action to -1 to open the gripper
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)

            # multiprocessing simulation
            for j in range(self.max_step_per_episode):
                agentview_rgb = obs["agentview_image"]
                eye_in_hand_rgb = obs["robot0_eye_in_hand_image"]

                joint_state = obs["robot0_joint_pos"]
                gripper_state = obs["robot0_gripper_qpos"]

                robot_states = np.concatenate([joint_state, gripper_state], axis=-1)

                # save_path = os.path.join("/home/i53/student/wang/OCIL/OCIL", f"{self.task_suite}", "images")
                # img = env.sim.render(camera_name="frontview", width=1280, height=800)[..., ::-1]
                # img = np.flip(img, axis=0)
                # cv2.imwrite(os.path.join(save_path, f"agentview_{context}_{context_ind[i]}_{j}.png"), img)

                # agentview_rgb = self.reverse_rgb_channels(agentview_rgb)
                # eye_in_hand_rgb = self.reverse_rgb_channels(eye_in_hand_rgb)

                obs_dict = {"agentview_rgb": agentview_rgb,
                            "eye_in_hand_rgb": eye_in_hand_rgb,
                            "lang_emb": task_emb,
                            "robot_states": robot_states}

                action = agent.predict(obs_dict)
                obs, r, done, _ = env.step(action)

                # if self.render:
                # env.render()

                if r == 1:
                    success[context, context_ind[i]] = r
                    # env.close()
                    break

            env.close()

    def get_task_embs(self, task_embs):
        self.task_embs = task_embs

    def test_agent(self, agent, agent_config, cpu_set=None, epoch=None):
        logging.info("Start testing agent")

        if cpu_set is None:
            num_cpu = self.n_cores
            cpu_set = [i for i in range(num_cpu)]
        else:
            num_cpu = len(cpu_set)

        print("there is {} cpus".format(num_cpu))

        if self.task_suite == "libero_90":
            num_tasks = 90
        else:
            num_tasks = 10

        success = torch.zeros([num_tasks, self.num_episode]).share_memory_()
        all_runs = num_tasks * self.num_episode
        ###################################################################
        # distribute every runs on cpu
        ###################################################################
        contexts = np.arange(num_tasks)
        contexts = np.repeat(contexts, self.num_episode)

        context_ind = np.arange(self.num_episode)
        context_ind = np.tile(context_ind, num_tasks)

        repeat_num = all_runs // num_cpu
        repeat_res = all_runs % num_cpu

        workload_array = np.ones([num_cpu], dtype=int)
        workload_array[:repeat_res] += repeat_num
        workload_array[repeat_res:] = repeat_num

        assert np.sum(workload_array) == all_runs

        ind_workload = np.cumsum(workload_array)
        ind_workload = np.concatenate([[0], ind_workload])
        ###################################################################
        ctx = mp.get_context('spawn')
        processes_list = []

        print("!!!!!!!!!!!!!!!!!!! agent scaler: ", agent.get_scaler)
        model_state_dict, scaler = agent.get_model_state
        shared_state_dict = {}
        for key, tensor in model_state_dict.items():
            shared_tensor = tensor.share_memory_()
            shared_state_dict[key] = shared_tensor

        for i in range(self.n_cores):
            p = ctx.Process(target=self.eval_agent,
                            kwargs={
                                "agent_config": agent_config,
                                "model_state_dict": shared_state_dict,
                                "scaler": scaler,
                                "contexts": contexts[ind_workload[i]:ind_workload[i + 1]],
                                "context_ind": context_ind[ind_workload[i]:ind_workload[i + 1]],
                                "success": success,
                                "pid": i,
                                "cpu_set": set(cpu_set[i:i + 1])
                            },
                            )
            p.start()
            processes_list.append(p)

        [p.join() for p in processes_list]

        success_rate = torch.mean(success, dim=-1)
        average_success = torch.mean(success_rate).item()

        print(f'success array {success.detach()}')

        custom_step = f"{epoch}_custom_step"
        wandb.define_metric(custom_step)
        wandb.define_metric(f"{epoch}_tasks_success", step_metric=custom_step)

        for num in range(num_tasks):
            log.info(f"Task {num}: {success_rate[num].item()}")

            wandb.log({custom_step: num,
                       f"{epoch}_tasks_success": success_rate[num].item()
                       })

        wandb.log({f"epoch{epoch}_average_success": average_success})
        log.info(f"Average success rate: {average_success}")
