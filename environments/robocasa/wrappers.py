from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import VectorWrapper
from robosuite.models.tasks.task import get_subtree_geom_ids_by_group
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)
from robosuite.wrappers import Wrapper as RoboSuiteWrapper
from tensordict import NonTensorData
from torch import Tensor

from environments.specs import (
    ActionSpec,
    CameraSpec,
    DataSpecs,
    DepthStream,
    PinholeCameraIntrinsic,
    RGBStream,
    TextSpec,
    space_to_spec,
)
from utils.math import convert_quat, normalize

if TYPE_CHECKING:
    from gymnasium.vector import VectorEnv

log = logging.getLogger(__name__)


class RoboCasaAdapter(
    gym.Wrapper[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray]
):

    # newer versions of gymnasium require render modes to be specified in metadata
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        env: gym.Env[dict[str, np.ndarray], np.ndarray],
        reduced_action_space: bool = False,
        render_cam_name: str | None = None,
        render_size: tuple[int, int] = (256, 256),
    ):
        super().__init__(env)

        if reduced_action_space:
            action_space = self.env.action_space
            low, high = action_space.low, action_space.high
            # In all robocasa environments, the action space is 12-dimensional
            # as the franka has a mobile platform.
            # We only control the first 7 dimensions, which correspond to the
            # end-effector and gripper.
            low, high = low[:7], high[:7]
            self.action_space = spaces.Box(low, high)
        self.reduced_action_space = reduced_action_space

        if self.unwrapped.use_camera_obs:
            camera_names = self.unwrapped.camera_names

            self.all_camera_keys = [
                key
                for key in self.observation_space.spaces.keys()
                if any(key.startswith(cam) for cam in camera_names)
            ]

            obs_spaces = self.observation_space.spaces.copy()
            for cam_name in camera_names:
                # trim singleton channel dimension from depth images
                if f"{cam_name}_depth" in obs_spaces:
                    obs_spaces[f"{cam_name}_depth"] = spaces.Box(
                        low=-float("inf"),
                        high=float("inf"),
                        shape=obs_spaces[f"{cam_name}_depth"].shape[:-1],
                        dtype=np.float32,
                    )

                if f"{cam_name}_segmentation_element" in obs_spaces:
                    obs_spaces[f"{cam_name}_segmentation_element"] = spaces.Box(
                        low=-float("inf"),
                        high=float("inf"),
                        shape=obs_spaces[f"{cam_name}_segmentation_element"].shape[:-1],
                        dtype=np.int64,
                    )

                # add obs space for extrinsic transform for each camera
                extrinsics_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float64
                )
                obs_spaces[f"{cam_name}_transform"] = extrinsics_space

            # add obs spaces for combined ee pose, base pose, and base to ee pose
            obs_spaces["ee_pose"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            )
            obs_spaces["base_pose"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            )
            obs_spaces["base_to_ee_pose"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            )
            obs_spaces["_action"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.action_space.shape,
                dtype=np.float32,
            )

            self.observation_space = spaces.Dict(obs_spaces)

        self.render_size = tuple(render_size)
        self.render_cam_name = render_cam_name or self.unwrapped.render_camera[0]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        info["success"] = False
        obs["_action"] = np.full(self.action_space.shape, np.NaN, dtype=np.float32)
        return self.observation(obs), info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        raw_action = action.copy()
        action = self.action(action)
        observation, reward, terminated, truncated, info = self.env.step(action)

        success = bool(self.unwrapped._check_success())
        info["success"] = success
        terminated = terminated or success

        observation["_action"] = raw_action

        return self.observation(observation), reward, terminated, truncated, info

    def action(self, action: np.ndarray) -> np.ndarray:
        """Returns a modified action before :meth:`step` is called."""
        if self.reduced_action_space:
            # In all robocasa environments, the action space is 12-dimensional
            # as the franka has a mobile platform.
            # We only control the first 7 dimensions, which correspond to the
            # end-effector and gripper.
            expanded_action = np.zeros(12, dtype=np.float32)
            expanded_action[-1] = -1
            expanded_action[:7] = action
            return expanded_action
        else:
            return action

    def observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Returns a modified observation."""
        if self.unwrapped.use_camera_obs:
            observation = self.cam_observation(observation)

        observation = self.quat_observation(observation)

        return observation

    @property
    def render_mode(self) -> str | None:
        if self.unwrapped.has_renderer:
            return "human"
        elif self.unwrapped.use_camera_obs or self.unwrapped.has_offscreen_renderer:
            return "rgb_array"
        else:
            return None

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            # TODO: cache rendered frames if both obs rendering and render
            # rendering share the same cameras?
            rendering = self.unwrapped.sim.render(
                camera_name=self.render_cam_name,
                height=self.render_size[0],
                width=self.render_size[1],
            )
            rendering = np.flip(rendering, axis=0)  # flip vertically
            return rendering
        else:
            return self.env.render()

    def quat_observation(
        self, observation: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        for key in observation.keys():
            if key.endswith("quat") or key.endswith("quat_site"):
                observation[key] = convert_quat(observation[key], "wxyz")

        ee_pose = np.concatenate(
            (observation["robot0_eef_pos"], observation["robot0_eef_quat_site"])
        )
        base_pose = np.concatenate(
            (observation["robot0_base_pos"], observation["robot0_base_quat"])
        )
        base_to_ee_pose = np.concatenate(
            (
                observation["robot0_base_to_eef_pos"],
                observation["robot0_base_to_eef_quat_site"],
            ),
        )

        observation["ee_pose"] = ee_pose
        observation["base_pose"] = base_pose
        observation["base_to_ee_pose"] = base_to_ee_pose

        return observation

    def cam_observation(
        self, observation: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        for key in self.all_camera_keys:
            value = observation[key]

            # flip rendered images vertically
            value = np.flip(value, axis=0)

            if key.endswith("_depth"):
                # remove singleton channel dimension
                value = value[..., 0]
                # convert from Mujoco simulation to depth in meters
                value = get_real_depth_map(self.unwrapped.sim, value)

            if key.endswith("_segmentation_element"):
                # remove singleton channel dimension
                value = value[..., 0]

            observation[key] = value

        for cam_name in self.unwrapped.camera_names:
            # add extrinsic transform for each camera to the obs dict
            extrinsics = get_camera_extrinsic_matrix(self.unwrapped.sim, cam_name)
            observation[f"{cam_name}_transform"] = extrinsics

        return observation

    def get_camera_intrinsic_matrix(self, camera_name: str) -> np.ndarray:
        camera_names = self.unwrapped.camera_names
        camera_idx = camera_names.index(camera_name)
        camera_height = self.unwrapped.camera_heights[camera_idx]
        camera_width = self.unwrapped.camera_widths[camera_idx]

        return get_camera_intrinsic_matrix(
            self.unwrapped.sim,
            camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
        ).astype(np.float32)


class RoboCasaSpecs(VectorWrapper):
    def __init__(self, env: VectorEnv[dict[str, Tensor], Tensor, Tensor]):
        super().__init__(env)

        wrapped_space = env.observation_space
        assert isinstance(wrapped_space, spaces.Dict)

        camera_names = env.unwrapped.get_attr("camera_names")[0]

        # group rendered observations by camera based on their keys
        camera_streams = {
            cam_name: [
                key
                for key in wrapped_space.spaces.keys()
                if key.startswith(cam_name) and not key.endswith("_transform")
            ]
            for cam_name in camera_names
        }

        all_camera_keys = [key for keys in camera_streams.values() for key in keys]

        obs_specs = {
            key: space_to_spec(space)
            for key, space in wrapped_space.items()
            if key not in all_camera_keys
        }

        for cam_name in camera_names:
            streams = {}

            key = f"{cam_name}_image"
            n_envs, time, height, width, n_channels = wrapped_space[key].shape
            assert n_envs == self.num_envs
            assert n_channels == 3
            streams["rgb"] = RGBStream(
                channels=3, channel_order="HWC", height=height, width=width
            )

            if (key := f"{cam_name}_depth") in wrapped_space.keys():
                assert (
                    self.num_envs,
                    time,
                    height,
                    width,
                ) == wrapped_space[key].shape
                streams["depth"] = DepthStream(height=height, width=width)

            intrinsics = self.unwrapped.call(
                "get_camera_intrinsic_matrix", camera_name=cam_name
            )[0]

            # dynamic pose provides complete transform to camera
            extrinsics = torch.eye(4, dtype=torch.float32)
            intrinsics = PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height, width
            )

            obs_specs[cam_name] = CameraSpec(
                streams=streams,
                time=time,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                # this transform is added by the RoboCasaAdapter wrapper to the obs
                dynamic_pose_obs_key=f"{cam_name}_transform",
            )

        action_spec = ActionSpec(action_dim=self.single_action_space.shape[-1])

        goal_specs = {"description": TextSpec()}

        self.camera_streams = camera_streams
        self.specs = DataSpecs(obs=obs_specs, action=action_spec, goal=goal_specs)

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options or {})
        return self.observations(obs), self.info(info)

    def step(
        self, actions: Tensor
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return self.observations(obs), reward, terminated, truncated, self.info(info)

    def observations(self, observations: dict[str, Tensor]) -> dict[str, Tensor]:
        for cam_name, stream_names in self.camera_streams.items():
            streams = {
                stream_name[len(cam_name) + 1 :]: observations.pop(stream_name)
                for stream_name in stream_names
            }

            # BackCompat: rename "image" stream to "rgb" for compatibility with
            # previous trained models, since the specs are frozen in the checkpoint
            if "image" in streams:
                streams["rgb"] = streams.pop("image")

            observations[cam_name] = streams

        return observations

    def info(self, info: dict[str, Any]) -> dict[str, Any]:
        ep_metas = self.unwrapped.call("get_ep_meta")
        descs = [ep_meta["lang"] for ep_meta in ep_metas]
        info["goal"] = {"description": descs}
        return info


class SegmentationWrapper(gym.Wrapper):
    def __init__(self, env, env_name: str):
        super().__init__(env)

        goal_mappings = {
            "TurnOnMicrowave": ["microwave"],
            "TurnOffMicrowave": ["microwave"],
            "CoffeePressButton": ["coffee_machine"],
            "CoffeeServeMug": ["obj", "counter"],
            "CoffeeSetupMug": ["obj", "coffee_machine"],
            "TurnSinkSpout": ["sink"],
            "TurnOnSinkFaucet": ["sink"],
            "TurnOffSinkFaucet": ["sink"],
            "CloseSingleDoor": ["door_fxtr"],
            "CloseDoubleDoor": ["door_fxtr"],
            "OpenDoubleDoor": ["door_fxtr"],
            "OpenSingleDoor": ["door_fxtr"],
            "CloseDrawer": ["drawer"],
            "OpenDrawer": ["drawer"],
            "TurnOffStove": ["stove"],
            "TurnOnStove": ["stove"],
            "PnPCounterToSink": ["obj", "sink"],
            "PnPCabToCounter": ["obj", "counter"],
            "PnPSinkToCounter": ["obj", "container"],
            "PnPMicrowaveToCounter": ["obj", "container"],
            "PnPCounterToStove": ["obj", "container"],
            "PnPCounterToCab": ["obj", "cab"],
            "PnPCounterToMicrowave": ["obj", "container"],
            "PnPStoveToCounter": ["obj", "container"],
        }

        self.local_goal_keys = goal_mappings.get(env_name, None)
        if self.local_goal_keys is None:
            raise ValueError(
                f"Unknown environment name {env_name} for SegmentationWrapper"
            )

        obs_spaces = self.observation_space.spaces.copy()
        segmentation_ids = list(self.unwrapped.obj_body_id.keys()) + list(
            self.unwrapped.fixtures_id.keys()
        )
        obs_spaces["segmentation_ids"] = spaces.Dict(
            {
                key: spaces.Sequence(
                    spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int64)
                )
                for key in segmentation_ids
                if key in self.local_goal_keys
            }
        )

        self.observation_space = spaces.Dict(obs_spaces)

        self.env = env

    def step(self, action):
        obs_dict, reward, done, truncated, info = self.env.step(action)

        obs_dict["segmentation_ids"] = self.get_segmentation_ids()

        return obs_dict, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        obs["segmentation_ids"] = self.get_segmentation_ids()

        return obs, info

    def reset_to(self, state):
        obs_dict = self.env.reset_to(state)

        obs_dict["segmentation_ids"] = NonTensorData(self.get_segmentation_ids())

        return obs_dict

    def get_segmentation_ids(self):
        # get the env at the deepest level
        env = self.unwrapped

        segmentation_ids = {}
        for body_key in env.obj_body_id.keys():
            if body_key not in self.local_goal_keys:
                continue
            obj_body_id = env.obj_body_id[body_key]
            segmentation_ids[body_key] = [
                geom_id
                for geom_id in range(env.sim.model.ngeom)
                if env.sim.model.geom_bodyid[geom_id] == obj_body_id and geom_id
            ]
            segmentation_ids[body_key] += get_subtree_geom_ids_by_group(
                env.sim.model, obj_body_id, group=1
            )

        for fixture_key in env.fixtures_id.keys():
            if fixture_key not in self.local_goal_keys:
                continue
            fixture_body_id = env.fixtures_id[fixture_key]
            segmentation_ids[fixture_key] = [
                geom_id
                for geom_id in range(env.sim.model.ngeom)
                if env.sim.model.geom_bodyid[geom_id] == fixture_body_id and geom_id
            ]
            segmentation_ids[fixture_key] += get_subtree_geom_ids_by_group(
                env.sim.model, fixture_body_id, group=1
            )

        return segmentation_ids


class GymWrapper(RoboSuiteWrapper, gym.Env):
    metadata = None
    render_mode = None
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.
        flatten_obs (bool):
            Whether to flatten the observation dictionary into a 1d array. Defaults to True.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None, flatten_obs=True):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join(
            [type(robot.robot_model).__name__ for robot in self.env.robots]
        )
        self.name = robots + "_" + type(self.env).__name__

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
                return spaces.Box(
                    low=low, high=high, shape=sample.shape, dtype=sample.dtype
                )

            self.observation_space = spaces.Dict(
                {key: get_box_space(obs[key]) for key in self.keys}
            )

        low, high = self.env.action_spec
        self.action_space = spaces.Box(low, high)

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
        obs = (
            self._flatten_obs(ob_dict)
            if self.flatten_obs
            else self._filter_obs(ob_dict)
        )
        return obs, {}

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
        obs = (
            self._flatten_obs(ob_dict)
            if self.flatten_obs
            else self._filter_obs(ob_dict)
        )
        return obs, reward, terminated, False, info

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
