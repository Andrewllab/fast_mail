import logging
import os
from typing import Any, Dict, NamedTuple, Optional, Tuple
from collections import deque

import torch
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist
import einops
import torch.optim as optim
import wandb

from .mdt.models.edm_diffusion.gc_sampling import *
from .mdt.models.edm_diffusion.utils import append_dims
from .mdt.models.perceptual_encoders.voltron_encoder import VoltronTokenEncoder
from .mdt.utils.ema import ExponentialMovingAverage

from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    for name, submodule in model.named_modules():
        # Adjusting the condition to capture the desired layers
        if '.' not in name or name.count('.') <= 10:  # Can be adjusted based on your model structure
            # Counting parameters including submodules
            submodule_params = sum(p.numel() for p in submodule.parameters())
            if submodule_params > 0:
                print(f"{name} - Total Params: {submodule_params}")


class MDTAgent(BaseAgent):
    def __init__(
            self,
            model: DictConfig,
            trainset: DictConfig,
            valset: DictConfig,
            train_batch_size,
            val_batch_size,
            num_workers,
            device: str,
            epoch: int,
            scale_data,
            decay: float,
            obs_seq_len,
            multistep: int = 10,
            masked_beta: float = 1,
            use_text_not_embedding: bool = False,
            ckpt_path=None,
            seed: int = 42,
            scaler_type: str = 'minmax'
    ):
        super(MDTAgent, self).__init__(model, trainset=trainset, valset=valset, train_batch_size=train_batch_size,
                                       val_batch_size=val_batch_size, num_workers=num_workers, device=device,
                                       epoch=epoch, scale_data=scale_data, scaler_type=scaler_type)

        self.ema_helper = ExponentialMovingAverage(self.model.parameters(), decay, self.device)

        # self.camera_types = self.trainset.cameras_type

        self.decay = decay

        self.optimizer, self.lr_scheduler = self.model.configure_optimizers()

        self.seed = seed

        self.modality_scope = "vis"

        self.masked_beta = masked_beta

        self.obs_seq_len = obs_seq_len

        # for inference
        self.rollout_step_counter = 0
        self.multistep = multistep
        self.latent_goal = None
        self.plan = None
        self.state_recons = False
        self.use_text_not_embedding = use_text_not_embedding

        self.bp_image_context = deque(maxlen=self.obs_seq_len)
        self.inhand_image_context = deque(maxlen=self.obs_seq_len)

        if ckpt_path is not None:
            self.load_pretrained_model(ckpt_path)

    def load_pretrained_model(self, weights_path: str, **kwargs) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """

        self.model.load_state_dict(torch.load(os.path.join(weights_path, "model_state_dict.pth")))
        self.ema_helper = ExponentialMovingAverage(self.model.parameters(), self.decay, self.device)

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """

        self.ema_helper.store(self.model.parameters())
        self.ema_helper.copy_to(self.model.parameters())
        torch.save(self.model.state_dict(), os.path.join(store_path, "model_state_dict.pth"))

        # self.ema_helper.restore(self.model.parameters())
        # torch.save(self.model.state_dict(), os.path.join(store_path, "non_ema_model_state_dict.pth"))

    def clip_extra_forward(self, perceptual_emb, latent_goal, actions, sigmas, noise):

        self.model.train()
        noised_input = actions + noise * append_dims(sigmas, actions.ndim)
        context = self.model.forward_context_only(perceptual_emb, noised_input, latent_goal, sigmas)
        return context

    def train_vision_agent(self):

        for num_epoch in tqdm(range(self.epoch)):

            epoch_loss = torch.tensor(0.0).to(self.device)

            for data in self.train_dataloader:
                bp_imgs, inhand_imgs, action, task_emb = data

                # for camera in obs_dict.keys():
                #     obs_dict[camera] = obs_dict[camera].to(self.device)
                #     obs_dict[camera] = obs_dict[camera][:, :self.obs_seq_len].contiguous()

                bp_imgs = bp_imgs.to(self.device)
                inhand_imgs = inhand_imgs.to(self.device)
                task_emb = task_emb.to(self.device)

                action = self.scaler.scale_output(action)
                # action = action[:, self.obs_seq_len - 1:, :].contiguous()

                bp_imgs = bp_imgs[:, :1].contiguous()
                inhand_imgs = inhand_imgs[:, :1].contiguous()

                state = (bp_imgs, inhand_imgs, task_emb)

                batch_loss = self.train_step(state, action)

                epoch_loss += batch_loss

            epoch_loss = epoch_loss / len(self.train_dataloader)

            wandb.log({"train_loss": epoch_loss.item()})
            log.info("Epoch {}: Mean train loss is {}".format(num_epoch, epoch_loss.item()))

        log.info("training done")
        self.store_model_weights(self.working_dir, sv_name='last_mdt.pth')

    def train_step(self, state: tuple, action: torch.Tensor, goal: Optional[torch.Tensor] = None) -> float:

        self.model.train()

        if goal is not None:
            goal = self.scaler.scale_input(goal)

        # Compute the loss.
        loss = self.model(state, goal, action=action, if_train=True)

        # Before the backward pass, zero all the network gradients
        self.optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to parameters
        loss.backward()
        # Calling the step function to update the parameters
        self.optimizer.step()
        self.lr_scheduler.step()

        self.ema_helper.update(self.model.parameters())
        return loss

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0

    @torch.no_grad()
    def predict(self, obs, goal=None):

        # imgs_seq = {}

        bp_image, inhand_image, task_emb = obs

        bp_image = torch.from_numpy(bp_image).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.
        inhand_image = torch.from_numpy(inhand_image).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.

        self.bp_image_context.append(bp_image)
        self.inhand_image_context.append(inhand_image)

        bp_image_seq = torch.stack(tuple(self.bp_image_context), dim=1)
        inhand_image_seq = torch.stack(tuple(self.inhand_image_context), dim=1)

        # for camera, data in obs.items():
        #     if 'rgb' not in camera:
        #         continue
        #
        #     obs[camera] = torch.from_numpy(data).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.
        #
        #     self.context_dict[camera].append(obs[camera])
        #
        #     imgs_seq[camera] = torch.stack(tuple(self.context_dict[camera]), dim=1)

        task_emb = task_emb.to(self.device).unsqueeze(0)

        input_state = (bp_image_seq, inhand_image_seq, task_emb)

        if self.rollout_step_counter % self.multistep == 0:

            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())

            self.model.eval()

            # predict action sequence
            pred_action_seq = self.model(input_state, goal)

            self.ema_helper.restore(self.model.parameters())

            pred_action_seq = self.scaler.inverse_scale_output(pred_action_seq)

            # return pred_action_seq.detach().cpu().numpy()

            self.pred_action_seq = pred_action_seq

        current_action = self.pred_action_seq[0, self.rollout_step_counter]

        if len(current_action.shape) == 2:
            current_action = einops.rearrange(current_action, 'b d -> b 1 d')

        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0

        return current_action.detach().cpu().numpy()

    def train_agent(self):
        pass