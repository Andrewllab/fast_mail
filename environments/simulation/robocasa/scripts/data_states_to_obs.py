"""
Script to extract observations from low-dimensional simulation states in a robocasa dataset.
Adapted from robomimic's dataset_states_to_obs.py script.
"""
import os
import json
from typing import OrderedDict
import h5py
import argparse
import numpy as np
from copy import deepcopy
import multiprocessing
import queue
import time
import traceback
import gymnasium as gym
from gymnasium import spaces
from robosuite.wrappers import Wrapper

import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils

# from robocasa.utils.env_utils import create_env
import robocasa.utils.robomimic.robomimic_tensor_utils as TensorUtils
import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
from tqdm import tqdm

# from robomimic.utils.log_utils import log_warning
from robosuite.utils import camera_utils


# taken from and adapted: https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/wrappers/gym_wrapper.py
class GymWrapper(Wrapper, gym.Env):
    metadata = None
    render_mode = "rgb_array"
    """
    Initializes a Gym wrapper for RoboCasa environments. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (RoboCasaEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.
        flatten_obs (bool):
            Whether to flatten the observation dictionary into a 1d array. Defaults to True.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, render_width, render_height, keys=None, flatten_obs=True):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # choose a default camera / size for video if not provided
        self._render_camera = (self.env.camera_names[0] if self.env.camera_names else "frontview")
        self._render_width = render_width
        self._render_height = render_height

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        obs = self.env.reset()

        # Whether to flatten the observation space
        self.flatten_obs: bool = flatten_obs

        if self.flatten_obs:
            flat_ob = self._flatten_obs(obs)
            self.obs_dim = flat_ob.size
            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low, high)
        else:

            def get_box_space(sample):
                """Util fn to obtain the space of a single numpy sample data"""
                if np.issubdtype(sample.dtype, np.integer):
                    low = np.iinfo(sample.dtype).min
                    high = np.iinfo(sample.dtype).max
                elif np.issubdtype(sample.dtype, np.inexact):
                    low = float("-inf")
                    high = float("inf")
                else:
                    raise ValueError()
                return spaces.Box(low=low, high=high, shape=sample.shape, dtype=sample.dtype)

            self.observation_space = spaces.Dict({key: get_box_space(obs[key]) for key in self.keys})

        low, high = self.env.action_spec
        self.action_space = spaces.Box(low, high)

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return "rgb_array"
    
    @property
    def unwrapped(self) -> str | None:
        return self.env

    def _process_observation(self, obs_dict):
        for key in obs_dict:
            if "image" in key or "segmentation" in key:
                obs_dict[key] = np.flip(obs_dict[key], axis=0)
            elif "depth" in key:
                obs_dict[key] = np.flip(obs_dict[key], axis=0)
                obs_dict[key] = self._depthimg2Meters(obs_dict[key])

        return obs_dict

    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def _depthimg2Meters(self, depth):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def _filter_obs(self, obs_dict) -> dict:
        """
        Filters keys of interest out of the observation dictionary, returning a filterd dictionary.
        """
        return {key: obs_dict[key] for key in self.keys if key in obs_dict}

    def render(self):
        """
        Return an rgb_array for RecordVideo. Uses offscreen renderer.
        """
        frame = self.env.sim.render(
            height=self._render_height,
            width=self._render_width,
            camera_name=self._render_camera,
        )
        # robocasa envs return upside-down frames -> flip vertically
        frame = np.flip(frame, axis=0)
        return frame
    
    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return observation instead of normal OrderedDict and optionally resets seed

        Returns:
            2-tuple:
                - (np.array) observations from the environment
                - (dict) an empty dictionary, as part of the standard return format
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()
        ob_dict = self._process_observation(ob_dict)
        # obs = self._flatten_obs(ob_dict) if self.flatten_obs else self._filter_obs(ob_dict)
        return ob_dict, {}

    def reset_to(self, state):
        ob_dict = self.env.reset_to(state)

        ob_dict = self._process_observation(ob_dict)
        # obs = self._flatten_obs(ob_dict) if self.flatten_obs else self._filter_obs(ob_dict)

        return ob_dict, {}

    def step(self, action):
        """
        Extends vanilla step() function call to return observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        ob_dict, reward, terminated, info = self.env.step(action)
        ob_dict = self._process_observation(ob_dict)

        # obs = self._flatten_obs(ob_dict) if self.flatten_obs else self._filter_obs(ob_dict)
        return ob_dict, reward, terminated, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()

    def close(self):
        """
        wrapper for calling underlying env close function
        """
        self.env.close()

    def _check_success(self):
        return self.env.is_success()



def record_cam_params(env, cam_names, W, H):
    """
    Returns {cam: {intrinsics (3x3), extrinsics (4x4 camera->world), width, height}}
    """
    sim = env.base_env.sim
    camera_params = {}
    for cam in cam_names:
        camera_intrinsics = camera_utils.get_camera_intrinsic_matrix(sim, cam, H, W)
        camera_extrinsics = camera_utils.get_camera_extrinsic_matrix(sim, cam)

        camera_params[cam] = {
            "intrinsics": camera_intrinsics.astype(np.float32),
            "extrinsics": camera_extrinsics.astype(np.float32),  # camera pose in WORLD FRAME
            "width": int(W),
            "height": int(H),
        }
    return camera_params



def get_camera_info(
    env,
    camera_names=None, 
    camera_height=84, 
    camera_width=84,
):
    """
    Helper function to get camera intrinsics and extrinsics for cameras being used for observations.
    """

    # TODO: make this function more general than just robosuite environments
    assert EnvUtils.is_robosuite_env(env=env)

    if camera_names is None:
        return None

    camera_info = dict()
    for cam_name in camera_names:
        K = env.get_camera_intrinsic_matrix(camera_name=cam_name, camera_height=camera_height, camera_width=camera_width)
        R = env.get_camera_extrinsic_matrix(camera_name=cam_name) # camera pose in world frame
        if "eye_in_hand" in cam_name:
            # convert extrinsic matrix to be relative to robot eef control frame
            assert cam_name.startswith("robot0")
            eef_site_name = env.base_env.robots[0].controller.eef_name
            eef_pos = np.array(env.base_env.sim.data.site_xpos[env.base_env.sim.model.site_name2id(eef_site_name)])
            eef_rot = np.array(env.base_env.sim.data.site_xmat[env.base_env.sim.model.site_name2id(eef_site_name)].reshape([3, 3]))
            eef_pose = np.zeros((4, 4)) # eef pose in world frame
            eef_pose[:3, :3] = eef_rot
            eef_pose[:3, 3] = eef_pos
            eef_pose[3, 3] = 1.0
            eef_pose_inv = np.zeros((4, 4))
            eef_pose_inv[:3, :3] = eef_pose[:3, :3].T
            eef_pose_inv[:3, 3] = -eef_pose_inv[:3, :3].dot(eef_pose[:3, 3])
            eef_pose_inv[3, 3] = 1.0
            R = R.dot(eef_pose_inv) # T_E^W * T_W^C = T_E^C
        camera_info[cam_name] = dict(
            intrinsics=K.tolist(),
            extrinsics=R.tolist(),
        )
    return camera_info


def extract_trajectory(
    env,
    initial_state,
    states,
    actions,
    done_mode,
    args,
    add_datagen_info=False,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a
            success state. If 1, done is 1 at the end of each trajectory.
            If 2, do both.
    """
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    env.reset_to(initial_state)

    # get updated ep meta in case it's been modified
    ep_meta = env.env.get_ep_meta()
    initial_state["ep_meta"] = json.dumps(ep_meta, indent=4)

    # extract camera parameters and prepare necessary lists/dicts for storing the information
    W = int(args.camera_width)
    H = int(args.camera_height)
    camera_names = list(args.camera_names)
    default_dynamic_camera_names = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    # split the cameras into static and moving
    # !!! IMPORTANT: the left/right view cameras that are supposed to be static ARE ACTUALLY NOT STATIC! 
    # The robot-arm is attached to a mobiled platform that is not controlled, but it can move as a post-effect of the robot-arm movements.
    # They are attached to the robot as they move as well when the mobile platform moves. 
    # This means that all cameras should be perceived as dynamic which requires recording camera extrinsics over time for all cameras.
    # dynamic_cam_names = [c for c in camera_names if ("eye_in_hand" in c) or ("wrist" in c)]
    dynamic_cam_names = [cam_name for cam_name in default_dynamic_camera_names]
    static_cam_names  = [c for c in camera_names if c not in dynamic_cam_names]
    # record static cameras once per episode (if any)
    # prepare per-step buffers for moving cameras
    dynamic_cam_logs = {name: {"intrinsics": [], "extrinsics": []} for name in dynamic_cam_names}
    static_cam_logs = {name: {"intrinsics": [], "extrinsics": []} for name in static_cam_names}

    traj = dict(
        obs=[],
        next_obs=[],
        rewards=[],
        dones=[],
        actions=np.array(actions),
        states=np.array(states),
        initial_state_dict=initial_state,
        datagen_info=[],
        dynamic_cameras=[],
        static_cameras=[],
    )

    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in tqdm(range(traj_len)):
        # print(f"STATIC CAMERA PARAMS: {record_cam_params(env, static_cam_names, W, H)}")
        obs = deepcopy(env.reset_to({"states": states[t]}))
        # flip images as RoboCasa's raw images are flipped, also convert depth to meters
        for key in obs:
            if "image" in key:
                obs[key] = np.flip(obs[key], axis=0)
            elif "depth" in key:
                obs[key] = np.flip(obs[key], axis=0)
                # https://github.com/ARISE-Initiative/robomimic/issues/203
                # https://github.com/ARISE-Initiative/robosuite/blob/9f28fb930ba1a07bd4a9a833f8a6b68e318aaf34/robosuite/utils/camera_utils.py#L106
                obs[key] = camera_utils.get_real_depth_map(env.base_env.sim, obs[key])

        if t == 0:
            # !!! IMPORTANT: record static camera parameters after environment reset !!! 
            if static_cam_names:
                static_camera_params = record_cam_params(env, static_cam_names, W, H)
                for name in static_cam_names:
                    static_cam_logs[name]["extrinsics"].append(static_camera_params[name]["extrinsics"])

        # each trajectory step record parameters for dynamically moving cameras
        if dynamic_cam_names:
            cam_parameters = record_cam_params(env, dynamic_cam_names, W, H)
            for name in dynamic_cam_names:
                # dynamic_cam_logs[name]["intrinsics"].append(cam_parameters[name]["intrinsics"])
                dynamic_cam_logs[name]["extrinsics"].append(cam_parameters[name]["extrinsics"])

        # extract datagen info
        if add_datagen_info:
            datagen_info = env.base_env.get_datagen_info(action=actions[t])
        else:
            datagen_info = {}

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
            # print(f"ENV DONE FIRST CASE: {done}")
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
            # print(f"ENV DONE SECOND CASE: {done}")
        done = int(done)

        # get the absolute action
        # action_abs = env.base_env.convert_rel_to_abs_action(actions[t])

        # collect transition
        traj["obs"].append(obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        traj["datagen_info"].append(datagen_info)


    # record intrinsics for all cameras only once as they don't change
    if dynamic_cam_names:
        cam_parameters = record_cam_params(env, dynamic_cam_names, W, H)
        for name in dynamic_cam_names:
            dynamic_cam_logs[name]["intrinsics"].append(cam_parameters[name]["intrinsics"])

    if static_cam_names:
        cam_parameters = record_cam_params(env, static_cam_names, W, H)
        for name in static_cam_names:
           static_cam_logs[name]["intrinsics"].append(cam_parameters[name]["intrinsics"])


    # # stack information about dynamic cameras
    # if static_cam_names:
    #     static_cam_params = {}
    #     for name in static_cam_names:
    #         # intrinsics_for_stacking = dynamic_cam_logs[name]["intrinsics"]
    #         # extrinsics_for_stacking = dynamic_cam_logs[name]["extrinsics"]
    #         # print(f"DYNAMIC CAM INTRINSICS SHAPE: {len(intrinsics_for_stacking), intrinsics_for_stacking[0].shape}")
    #         # print(f"DYNAMIC CAM EXRINSICS SHAPE: {len(extrinsics_for_stacking), extrinsics_for_stacking[0].shape}")
    #         intrinsics = np.stack(static_cam_logs[name]["intrinsics"], axis=0).astype(np.float32)  # (T,3,3)
    #         extrinsics = np.stack(static_cam_logs[name]["extrinsics"], axis=0).astype(np.float32)  # (T,4,4)
    #         static_cam_params[name] = {
    #             "intrinsics": intrinsics,
    #             "extrinsics": extrinsics,
    #             "width": W,
    #             "height": H,
    #         }
    #     traj["static_cameras"].append(static_cam_params)

    # stack information about dynamic cameras
    if dynamic_cam_names:
        dynamic_cam_params = {}
        for name in dynamic_cam_names:
            # intrinsics_for_stacking = dynamic_cam_logs[name]["intrinsics"]
            # extrinsics_for_stacking = dynamic_cam_logs[name]["extrinsics"]
            # print(f"DYNAMIC CAM INTRINSICS SHAPE: {len(intrinsics_for_stacking), intrinsics_for_stacking[0].shape}")
            # print(f"DYNAMIC CAM EXRINSICS SHAPE: {len(extrinsics_for_stacking), extrinsics_for_stacking[0].shape}")
            intrinsics = np.stack(dynamic_cam_logs[name]["intrinsics"], axis=0).astype(np.float32)  # (T,3,3)
            extrinsics = np.stack(dynamic_cam_logs[name]["extrinsics"], axis=0).astype(np.float32)  # (T,4,4)
            dynamic_cam_params[name] = {
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "width": W,
                "height": H,
            }
        traj["dynamic_cameras"].append(dynamic_cam_params)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    # traj_obs = traj["obs"]
    # print(f"TRAJECTORY OBSERVATIONS: {traj_obs}")
    # print(f"TRAJECTORY OBSERVATIONS: {traj_obs}")
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["datagen_info"] = TensorUtils.list_of_flat_dict_to_dict_of_list(
        traj["datagen_info"]
    )

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                if isinstance(traj[k][kp], dict):
                    traj[k][kp] = TensorUtils.list_of_flat_dict_to_dict_of_list(
                        traj[k][kp]
                    )
                    for kpp in traj[k][kp]:
                        traj[k][kp][kpp] = np.array(traj[k][kp][kpp])
                else:
                    traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


""" The process that writes over the generated files to memory """


def write_traj_to_file(
    args, output_path, total_samples, total_run, processes, mul_queue
):
    f = h5py.File(args.dataset, "r")
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    start_time = time.time()
    num_processed = 0

    try:
        while (total_run.value < (processes)) or not mul_queue.empty():
            if not mul_queue.empty():
                num_processed = num_processed + 1
                item = mul_queue.get()
                ep = item[0]
                traj = item[1]
                process_num = item[2]
                try:
                    ep_data_grp = data_grp.create_group(ep)
                    ep_data_grp.create_dataset(
                        "actions", data=np.array(traj["actions"])
                    )
                    ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
                    ep_data_grp.create_dataset(
                        "rewards", data=np.array(traj["rewards"])
                    )
                    ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
                    # ep_data_grp.create_dataset(
                    #     "actions_abs", data=np.array(traj["actions_abs"])
                    # )
                    for k in traj["obs"]:
                        if isinstance(traj["obs"][k], OrderedDict):
                            for kp in traj["obs"][k]:
                                value = traj["obs"][k]
                                print(f"Create dataset for: {kp, value}")
                                if args.no_compress:
                                    ep_data_grp.create_dataset(
                                        "obs/{}/{}".format(k, kp),
                                        data=np.array(traj["obs"][k][kp]),
                                    )
                                else:
                                    ep_data_grp.create_dataset(
                                        "obs/{}/{}".format(k, kp),
                                        data=np.array(traj["obs"][k][kp]),
                                        compression="gzip",
                                    )
                            continue
                        if args.no_compress:
                            ep_data_grp.create_dataset(
                                "obs/{}".format(k), data=np.array(traj["obs"][k])
                            )
                        else:
                            ep_data_grp.create_dataset(
                                "obs/{}".format(k),
                                data=np.array(traj["obs"][k]),
                                compression="gzip",
                            )
                        if args.include_next_obs:
                            if args.no_compress:
                                ep_data_grp.create_dataset(
                                    "next_obs/{}".format(k),
                                    data=np.array(traj["next_obs"][k]),
                                )
                            else:
                                ep_data_grp.create_dataset(
                                    "next_obs/{}".format(k),
                                    data=np.array(traj["next_obs"][k]),
                                    compression="gzip",
                                )

                    # static_cameras = traj["static_cameras"]
                    # dynamic_cameras = traj["dynamic_cameras"]
                    # assert "static_cameras" in traj
                    # assert "dynamic_cameras" in traj
                    # print(f"CAMERA PARAMETERS: STATIC: {static_cameras}, DYNAMIC: {dynamic_cameras}")
                    if "static_cameras" in traj:
                        # stat_cam_key = traj["static_cameras"]
                        # print(f"KOMMST DU BEI STATIC CAMERAS: {stat_cam_key}")
                        try:
                            for camera_name in traj["static_cameras"][0].keys():
                                # subkey_values = traj["static_cameras"][k]
                                # print(f"writing static camera parameters: {k}, {subkey_values}")
                                for camera_data_key, camera_data_value in traj["static_cameras"][0][camera_name].items():
                                    ep_data_grp.create_dataset(
                                        "camera_params/static/{}/{}".format(camera_name, camera_data_key),
                                        data=camera_data_value,
                                    )
                        except IndexError as e:
                            static_cam_info = traj["static_cameras"]
                            print(f"Attempt to store information about static cameras but there are no static cameras: {static_cam_info}.")

                    if "dynamic_cameras" in traj:
                        # dyn_cam = traj["static_cameras"]
                        # print(f"KOMMST DU BEI DYNAMIC CAMERAS: {dyn_cam}")
                        for camera_name in traj["dynamic_cameras"][0].keys():
                            for camera_data_key, camera_data_value in traj["dynamic_cameras"][0][camera_name].items(): 
                                ep_data_grp.create_dataset(
                                    "camera_params/dynamic/{}/{}".format(camera_name, camera_data_key),
                                    data=camera_data_value,
                                )

                    if "datagen_info" in traj:
                        for k in traj["datagen_info"]:
                            ep_data_grp.create_dataset(
                                "datagen_info/{}".format(k),
                                data=np.array(traj["datagen_info"][k]),
                            )

                    # copy action dict (if applicable)
                    if "data/{}/action_dict".format(ep) in f:
                        action_dict = f["data/{}/action_dict".format(ep)]
                        for k in action_dict:
                            ep_data_grp.create_dataset(
                                "action_dict/{}".format(k),
                                data=np.array(action_dict[k][()]),
                            )

                    # episode metadata
                    ep_data_grp.attrs["model_file"] = traj["initial_state_dict"][
                        "model"
                    ]  # model xml for this episode
                    ep_data_grp.attrs["ep_meta"] = traj["initial_state_dict"][
                        "ep_meta"
                    ]  # ep meta data for this episode
                    # if "ep_meta" in f["data/{}".format(ep)].attrs:
                    #     ep_data_grp.attrs["ep_meta"] = f["data/{}".format(ep)].attrs["ep_meta"]
                    ep_data_grp.attrs["num_samples"] = traj["actions"].shape[
                        0
                    ]  # number of transitions in this episode

                    total_samples.value += traj["actions"].shape[0]
                except Exception as e:
                    print("++" * 50)
                    print(
                        f"Error at Process {process_num} on episode {ep} with \n\n {e}"
                    )
                    print("++" * 50)
                    raise Exception("Write out to file has failed")
                print(
                    "ep {}: wrote {} transitions to group {} at process {} with {} finished. Datagen rate: {:.2f} sec/demo".format(
                        num_processed,
                        ep_data_grp.attrs["num_samples"],
                        ep,
                        process_num,
                        total_run.value,
                        (time.time() - start_time) / num_processed,
                    )
                )
    except KeyboardInterrupt:
        print("Control C pressed. Closing File and ending \n\n\n\n\n\n\n")

    # we don't use masks
    # if "mask" in f:
    #     f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples.value
    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    if args.generative_textures:
        env_meta["env_kwargs"]["generative_textures"] = "100p"
    if args.randomize_cameras:
        env_meta["env_kwargs"]["randomize_cameras"] = True
    print("total processes end {}".format(total_run.value))
    # data_grp.attrs["env_args"] = json.dumps(
    #     env.serialize(), indent=4
    # )  # environment info
    print("Wrote {} total samples to {}".format(total_samples.value, output_path))

    f_out.close()
    f.close()

    DatasetUtils.extract_action_dict(dataset=output_path)
    # DatasetUtils.make_demo_ids_contiguous(dataset=output_path)
    for num_demos in [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        75,
        80,
        90,
        100,
        125,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        1500,
        2000,
        2500,
        3000,
        4000,
        5000,
        10000,
    ]:
        DatasetUtils.filter_dataset_size(
            output_path,
            num_demos=num_demos,
        )

    print("Writing has finished")

    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    return


# runs multiple trajectory. If there has been an unrecoverable error, the system puts the current work back into the queue and exits
def extract_multiple_trajectories(
    process_num, current_work_array, work_queue, lock, args2, num_finished, mul_queue
):
    try:
        extract_multiple_trajectories_with_error(
            process_num, current_work_array, work_queue, lock, args2, mul_queue
        )
    except Exception as e:
        work_queue.put(current_work_array[process_num])
        print("*>*" * 50)
        print("Error process num {}:".format(process_num))
        print(e)
        print(traceback.format_exc())
        print("*>*" * 50)
        print()

    num_finished.value = num_finished.value + 1


def retrieve_new_index(process_num, current_work_array, work_queue, lock):
    with lock:
        if work_queue.empty():
            return -1
        try:
            tmp = work_queue.get(False)
            current_work_array[process_num] = tmp
            return tmp
        except queue.Empty:
            return -1


def extract_multiple_trajectories_with_error(
    process_num, current_work_array, work_queue, lock, args, mul_queue
):
    # create environment to use for data processing

    if args.add_datagen_info:
        import mimicgen.utils.file_utils as MG_FileUtils

        env_meta = MG_FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    else:
        env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    if args.generative_textures:
        env_meta["env_kwargs"]["generative_textures"] = "100p"
    if args.randomize_cameras:
        env_meta["env_kwargs"]["randomize_cameras"] = True
    # env = create_env_with_wrappers(env_meta["env_name"], args)


    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    if args.generative_textures:
        env_meta["env_kwargs"]["generative_textures"] = "100p"
    if args.randomize_cameras:
        env_meta["env_kwargs"]["randomize_cameras"] = True
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=args.shaped,
    )
    # env = GymWrapper(env, render_width=args.camera_width, render_height=args.camera_height)
    start_time = time.time()

    # print("==== Using environment with the following metadata ====")
    # print(json.dumps(env.serialize(), indent=4))
    # print("")

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[: args.n]

    ind = retrieve_new_index(process_num, current_work_array, work_queue, lock)
    while (not work_queue.empty()) and (ind != -1):
        try:
            # print("Running {} index".format(ind))
            ep = demos[ind]

            # prepare initial state to reload from
            states = f["data/{}/states".format(ep)][()]
            initial_state = dict(states=states[0])
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
            initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get(
                "ep_meta", None
            )

            # extract obs, rewards, dones
            actions = f["data/{}/actions".format(ep)][()]

            traj = extract_trajectory(
                env=env,
                initial_state=initial_state,
                states=states,
                actions=actions,
                done_mode=args.done_mode,
                add_datagen_info=args.add_datagen_info,
                args=args,
            )

            # maybe copy reward or done signal from source file
            if args.copy_rewards:
                traj["rewards"] = f["data/{}/rewards".format(ep)][()]
            if args.copy_dones:
                traj["dones"] = f["data/{}/dones".format(ep)][()]

            ep_grp = f["data/{}".format(ep)]

            states = ep_grp["states"][()]
            initial_state = dict(states=states[0])
            initial_state["model"] = ep_grp.attrs["model_file"]
            initial_state["ep_meta"] = ep_grp.attrs.get("ep_meta", None)

            # store transitions

            # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
            #            consistent as well
            # print("(process {}): ADD TO QUEUE index {}".format(process_num, ind))
            mul_queue.put([ep, traj, process_num])

            ind = retrieve_new_index(process_num, current_work_array, work_queue, lock)
        except Exception as e:
            print("_" * 50)
            print("Process {}:".format(process_num))
            print("Error processing demo index {}: {}".format(ind, e))
            print(traceback.format_exc())
            print("_" * 50)
            del env
            # env = create_env_with_wrappers(env_meta["env_name"], args)
            env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
            if args.generative_textures:
                env_meta["env_kwargs"]["generative_textures"] = "100p"
            if args.randomize_cameras:
                env_meta["env_kwargs"]["randomize_cameras"] = True
            env = EnvUtils.create_env_for_data_processing(
                env_meta=env_meta,
                camera_names=args.camera_names,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
                reward_shaping=args.shaped,
            )
            # env = GymWrapper(env, render_width=args.cameras_width, render_height=args.cameras_height)

    f.close()
    print("Process {} finished".format(process_num))


# def create_env_with_wrappers(env_name, args):
#     base_env = create_env(
#         env_name=env_name,
#         camera_names=args.camera_names,
#         camera_heights=args.camera_height,
#         camera_widths=args.camera_width,
#         camera_depths=True,
#     )
#     obs = base_env.reset()
#     base_env.reset_to(obs["state"])

#     env = GymWrapper(base_env, render_height=args.camera_height, render_width=args.camera_width)

#     return env


def dataset_states_to_obs_multiprocessing(args):
    # create environment to use for data processing

    # output file in same directory as input file
    output_name = args.output_name
    if output_name is None:
        if len(args.camera_names) == 0:
            output_name = os.path.basename(args.dataset)[:-5] + "_ld.hdf5"
        else:
            image_suffix = str(args.camera_width)
            image_suffix = (
                image_suffix + "_randcams" if args.randomize_cameras else image_suffix
            )
            if args.generative_textures:
                output_name = os.path.basename(args.dataset)[
                    :-5
                ] + "_gentex_im{}.hdf5".format(image_suffix)
            else:
                output_name = os.path.basename(args.dataset)[:-5] + "_im{}.hdf5".format(
                    image_suffix
                )

    output_path = os.path.join(os.path.dirname(args.dataset), output_name)

    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))

    f = h5py.File(args.dataset, "r")
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    if args.n is not None:
        demos = demos[: args.n]

    num_demos = len(demos)
    f.close()

    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    num_processes = args.num_procs

    index = multiprocessing.Value("i", 0)
    lock = multiprocessing.Lock()
    total_samples_shared = multiprocessing.Value("i", 0)
    num_finished = multiprocessing.Value("i", 0)
    mul_queue = multiprocessing.Queue()
    work_queue = multiprocessing.Queue()
    for index in range(num_demos):
        work_queue.put(index)
    current_work_array = multiprocessing.Array("i", num_processes)
    processes = []
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=extract_multiple_trajectories,
            args=(
                i,
                current_work_array,
                work_queue,
                lock,
                args,
                num_finished,
                mul_queue,
            ),
        )
        processes.append(process)

    process1 = multiprocessing.Process(
        target=write_traj_to_file,
        args=(
            args,
            output_path,
            total_samples_shared,
            num_finished,
            num_processes,
            mul_queue,
        ),
    )
    processes.append(process1)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    print("Finished Multiprocessing")
    return


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        help="name of output hdf5 dataset",
    )

    parser.add_argument(
        "--filter_key",
        type=str,
        help="filter key for input dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    # flag for reward shaping
    parser.add_argument(
        "--shaped",
        action="store_true",
        help="(optional) use shaped rewards",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=224,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=224,
        help="(optional) width of image observations",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards",
        action="store_true",
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones",
        action="store_true",
        help="(optional) copy dones from source file instead of inferring them",
    )

    # flag to include next obs in dataset
    parser.add_argument(
        "--include-next-obs",
        action="store_true",
        help="(optional) include next obs in dataset",
    )

    # flag to disable compressing observations with gzip option in hdf5
    parser.add_argument(
        "--no_compress",
        action="store_true",
        help="(optional) disable compressing observations with gzip option in hdf5",
    )

    parser.add_argument(
        "--num_procs",
        type=int,
        default=5,
        help="number of parallel processes for extracting image obs",
    )

    parser.add_argument(
        "--add_datagen_info",
        action="store_true",
        help="(optional) add datagen info (used for mimicgen)",
    )

    parser.add_argument("--generative_textures", action="store_true")

    parser.add_argument("--randomize_cameras", action="store_true")

    args = parser.parse_args()
    dataset_states_to_obs_multiprocessing(args)
