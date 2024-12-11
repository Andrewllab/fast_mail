import torch
import torch.nn as nn

from omegaconf import DictConfig
import hydra


class BC_Policy(nn.Module):

    def __init__(self,
                 backbones: DictConfig,
                 obs_encoders: DictConfig,
                 if_robot_states: bool = False,
                 if_film_condition: bool = False,
                 device: str = 'cpu',
                 state_dim: int = 7,
                 latent_dim: int = 64):

        super(BC_Policy, self).__init__()

        self.img_encoder = hydra.utils.instantiate(obs_encoders).to(device)
        self.model = hydra.utils.instantiate(backbones).to(device)

        self.if_robot_states = if_robot_states
        self.if_film_condition = if_film_condition
        self.state_emb = nn.Linear(state_dim, latent_dim)

    def compute_input_embeddings(self, obs_dict):
        """
        Compute the required embeddings for the visual ones and the latent goal.
        """

        latent_goal = obs_dict['lang_emb']

        # print(f"the shape of this dict is {obs_dict[list(obs_dict.keys())[0]].shape}")
        B, T, C, H, W = obs_dict[list(obs_dict.keys())[0]].shape

        for camera in obs_dict.keys():
            if 'rgb' not in camera:
                continue
            # print(obs_dict[camera].shape)
            obs_dict[camera] = obs_dict[camera].view(B * T, C, H, W)
        # print(self.if_film_condition)
        if self.if_film_condition:
            perceptual_emb = self.img_encoder(obs_dict, latent_goal)
        else:
            # obs_dict is a dict with two images and one lang: images are [64,3,256,256]
            perceptual_emb = self.img_encoder(obs_dict)

        if self.if_robot_states and "robot_states" in obs_dict.keys():
            robot_states = obs_dict['robot_states']
            robot_states = self.state_emb(robot_states)

            perceptual_emb = torch.cat([perceptual_emb, robot_states], dim=1)

        return perceptual_emb, latent_goal

    def forward(self, inputs, return_encoder_embedding=False):

        # with torch.no_grad():
        perceptual_emb, latent_goal = self.compute_input_embeddings(inputs)
        # shape of perceptural_emb is torch.Size([64, 1, 256])
        # make prediction
        pred = self.model(
            perceptual_emb,
            latent_goal,
            return_encoder_embedding=return_encoder_embedding
        )

        return pred

    def get_params(self):
        return self.parameters()
