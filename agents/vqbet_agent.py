import os
import hydra
from omegaconf import DictConfig
from agents.base_agent import BaseAgent
# vq vae
# class VqVAE:
#     def __init__(self, device: str = "cpu"):
#         pass

#     def pretrain(self):
#         pass

#     def set_scaler(self, scaler):
#         pass

#     def save_model(self):
#         pass

#     def load_model(self):
#         pass

#     @property
#     def get_model_state(self):
#         pass

#     def recover_model_state(self, model_state):
#         pass


# vq agent
class VqBetAgent(BaseAgent):
    def __init__(
        self,
        model: DictConfig,
        vqvae: DictConfig,
        obs_encoders: DictConfig,
        language_encoders: DictConfig,
        optimization: DictConfig,
        action_seq_size: int,
        if_robot_states: bool = False,
        if_film_condition: bool = False,
        device: str = "cpu",
        state_dim: int = 7,
        latent_dim: int = 64,
        multistep: int = 10,
    ):
        super().__init__(
            model=model,
            obs_encoders=obs_encoders,
            language_encoders=language_encoders,
            device=device,
            state_dim=state_dim,
            latent_dim=latent_dim,
            multistep=multistep
        )

        # todo: after pretrain, freeze gradient
        print("VQVAE Configuration:")
        print(vqvae)
        
        self.vqvae = hydra.utils.instantiate(vqvae)
        self.optimizer_config = optimization
        self.use_lr_scheduler = False


        # if not pretrain:
        #     self.vqvae.load_model(vqvae_model_state_path)
        #     #freeze gradient (no gradient for vqvae)
        #     for param in self.vqvae.parameters():
        #         param.requires_grad = False

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.optimizer_config, params=self.parameters()
        )
        return optimizer


    def set_scaler(self, scaler):
        self.scaler = scaler
        self.vqvae.set_scaler(scaler)


    def forward(self, actions):
        pass
    