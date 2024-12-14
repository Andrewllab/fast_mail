import abc
import os
import logging

import torch
import torch.nn as nn
from omegaconf import DictConfig
import hydra

import wandb

from agents.utils.scaler import Scaler, ActionScaler, MinMaxScaler

# A logger for this file
log = logging.getLogger(__name__)


class BaseAgent(nn.Module, abc.ABC):

    def __init__(
            self,
            device: str = 'cpu',
    ):

        super(BaseAgent, self).__init__()

        self.device = device
        self.working_dir = os.getcwd()

    def get_scaler(self, scaler):
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
        Method to load a pretrained model weights inside self.model
        """

        if sv_name is None:
            self.model.load_state_dict(torch.load(os.path.join(weights_path, "model_state_dict.pth")))
        else:
            self.model.load_state_dict(torch.load(os.path.join(weights_path, sv_name)))
        log.info('Loaded pre-trained model parameters')

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """

        if sv_name is None:
            torch.save(self.model.state_dict(), os.path.join(store_path, "model_state_dict.pth"))
        else:
            torch.save(self.model.state_dict(), os.path.join(store_path, sv_name))

    def get_params(self):

        total_params = sum(p.numel() for p in self.parameters())

        wandb.log(
            {
                "model parameters": total_params
            }
        )

        log.info("The model has a total amount of {} parameters".format(total_params))
