import gym

from utils.point_cloud.sampling.base_pc_sampler import BasePointCloudSampler


class PointCloudSamplingWrapper(gym.ObservationWrapper):
    def __init__(self, env, pc_sampler: BasePointCloudSampler, num_points: int):
        super().__init__(env)
        self.pc_sampler = pc_sampler
        self.num_points = num_points
        
    def observation(self, obs_dict):
        obs_dict["sampled_point_cloud"] = self.pc_sampler.sample(obs_dict["point_cloud"], self.num_points)
        return obs_dict