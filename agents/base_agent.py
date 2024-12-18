import abc
import os
import logging

import torch
import torch.nn as nn
from omegaconf import DictConfig
import hydra
import pickle
import wandb

from agents.utils.scaler import Scaler, ActionScaler, MinMaxScaler

# A logger for this file
log = logging.getLogger(__name__)


class BaseAgent(nn.Module, abc.ABC):

    def __init__(
            self,
            model: DictConfig,
            obs_encoders: DictConfig,
            device: str,
            state_dim: int,
            latent_dim: int
    ):
        super().__init__()

        self.device = device
        self.working_dir = os.getcwd()
        self.scaler = None
        
        # Initialize model and encoder
        self.img_encoder = hydra.utils.instantiate(obs_encoders).to(device)
        self.model = hydra.utils.instantiate(model).to(device)
        self.state_emb = nn.Linear(state_dim, latent_dim)

    def set_scaler(self, scaler):
        self.scaler = scaler

    @abc.abstractmethod
    def compute_input_embeddings(self, obs_dict):
        """
        Compute the required embeddings for the visual ones and the latent goal.
        """
        pass

    @abc.abstractmethod
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """
        Method for predicting one step with input data
        """
        pass

    @abc.abstractmethod
    def reset(self) -> torch.Tensor:
        """
        Method for resetting the agent
        """
        pass

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        """
        Method to load pretrained weights for the entire agent
        """
        path = os.path.join(weights_path, "model_state_dict.pth" if sv_name is None else f"{sv_name}.pth")
        self.load_state_dict(torch.load(path, weights_only=True))
        log.info('Loaded pre-trained model')

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the weights of the entire agent
        """
        path = os.path.join(store_path, "model_state_dict.pth" if sv_name is None else f"{sv_name}.pth")
        torch.save(self.state_dict(), path)
        log.info(f'Model saved to: {store_path}')

    def store_model_scaler(self, store_path: str, sv_name=None) -> None:
        """
        Store the model scaler inside the store path as model_scaler.pkl
        """        
        save_path = os.path.join(store_path, "model_scaler.pkl" if sv_name is None else sv_name)
        with open(save_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        log.info(f'Model scaler saved to: {save_path}')

    def load_model_scaler(self, weights_path: str, sv_name=None) -> None:
        """
        Load the model scaler from the weights path
        """
        if sv_name is None:
            sv_name = "model_scaler.pkl"
        
        with open(os.path.join(weights_path, sv_name), 'rb') as f:
            self.scaler = pickle.load(f)
        log.info('Loaded model scaler')

    def get_params(self):

        total_params = sum(p.numel() for p in self.parameters())

        wandb.log(
            {
                "model parameters": total_params
            }
        )

        log.info("The model has a total amount of {} parameters".format(total_params))

    @property
    def get_model_state_dict(self) -> dict:
        return self.state_dict()

    @property
    def get_scaler(self) -> Scaler:
        if self.scaler is None:
            raise AttributeError("Scaler has not been set. Use set_scaler() first.")
        return self.scaler
    
    @property
    def get_model_state(self) -> tuple[dict, Scaler]:
        if self.scaler is None:
            raise AttributeError("Scaler has not been set. Use set_scaler() first.")
        return (
            self.state_dict(),
            self.get_scaler
        )
    
    def recover_model_state(self, model_state, scaler):
        self.load_state_dict(model_state)
        self.set_scaler(scaler)

    