import gym
import numpy as np
from robosuite.utils.transform_utils import (
    axisangle2quat,
    quat2axisangle,
    quat2mat,
    euler2mat,
    mat2quat,
)


class RobosuiteWrapper(gym.Wrapper):
    """
    This wrapper does the following changes to the action and observation:
    - Flips the image and segmentation observations
    - Converts the depth observation from depth image to meters
    - Rotates the end effector rotation by -90 degrees around the z-axis. 
      This ensures that end effector rotation in the observation is the same as the rotation in the action given to this method.
      See: https://github.com/ARISE-Initiative/robosuite/issues/337#issuecomment-1153684026
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        new_action = action.copy()

        rotation = new_action[3:6]
        rotation_mat = quat2mat(axisangle2quat(rotation))
        rotation_mat = rotation_mat @ euler2mat(np.array([0, 0, -np.pi / 2]))
        new_action[3:6] = quat2axisangle(mat2quat(rotation_mat))

        obs_dict, reward, done, info = self.env.step(new_action)

        obs_dict = self.process_observation(obs_dict)

        return obs_dict, reward, done, info

    def reset(self):
        obs_dict = self.env.reset()

        obs_dict = self.process_observation(obs_dict)

        return obs_dict

    def reset_to(self, state):
        obs_dict = self.env.reset_to(state)

        obs_dict = self.process_observation(obs_dict)

        return obs_dict

    def process_observation(self, obs_dict):
        for key in obs_dict:
            if "image" in key or "segmentation" in key:
                obs_dict[key] = np.flip(obs_dict[key], axis=0)
            elif "depth" in key:
                obs_dict[key] = np.flip(obs_dict[key], axis=0)
                obs_dict[key] = self.depthimg2Meters(obs_dict[key])

        return obs_dict

    def _check_success(self):
        return self.env._check_success()

    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def depthimg2Meters(self, depth):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image
