import os
import hydra

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
import hydra
from typing import Optional
from agents.base_agent import BaseAgent

from agents.models.vqbet.utils import MLP


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


# vqbet agent
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
        action_dim: int = 7,
        gamma: float = 2.0,
        offset_loss_multiplier: float = 10,
        secondary_code_multiplier: float = 2,
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

        self.if_robot_states = if_robot_states
        self.if_film_condition = if_film_condition
        self.eval_model_name = "eval_best_vqbet.pth"
        self.last_model_name = "last_vqbet.pth"

        self.vqvae = hydra.utils.instantiate(vqvae)
        self.optimizer_config = optimization
        self.use_lr_scheduler = False

        # vqbet
        self._G = vqvae.vqvae_groups # G(number of groups)
        self._C = vqvae.vqvae_n_embed # C(number of code integers)
        self._D = vqvae.n_latent_dims  # D(latent dims)
        
        self._act_dim = action_dim
        self._act_seq_len = action_seq_size
        # self._act_window_size = act_window_size

        self._offset_loss_multiplier = offset_loss_multiplier
        self._secondary_code_multiplier = secondary_code_multiplier
        self._criterion = FocalLoss(gamma=gamma)
        self._map_to_cbet_preds_bin = MLP( 
                in_channels=self._act_dim * self._act_seq_len,
                hidden_channels=[512, 512, self._G * self._C],
            ).to(self.device) 
            
        
        self._map_to_cbet_preds_offset = MLP(
            in_channels=self._act_dim * self._act_seq_len,
            hidden_channels=[
                512,
                512,
                self._G * self._C * (self._act_dim * self._act_seq_len),
            ],
        ).to(self.device)

        # Print parameter sizes
        model_params = sum(p.numel() for p in self.model.parameters())
        bin_params = sum(p.numel() for p in self._map_to_cbet_preds_bin.parameters())
        offset_params = sum(p.numel() for p in self._map_to_cbet_preds_offset.parameters())
        
        print(f"Model parameters: {model_params:,}")
        print(f"CBET bin predictor parameters: {bin_params:,}")
        print(f"CBET offset predictor parameters: {offset_params:,}")

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.optimizer_config, params=self.parameters()
        )
        return optimizer

    def set_scaler(self, scaler):
        self.scaler = scaler
        self.vqvae.set_scaler(scaler)

    def forward(self, obs_dict, actions=None):
        perceptual_emb, latent_goal = self.compute_input_embeddings(obs_dict)

        backbone_output = self.model(perceptual_emb, latent_goal)

        if self.training and actions is not None:
            predicted_action, loss, loss_dict =self._predict_cbet(backbone_output, actions)
            return loss
        else:
            predicted_action, loss, loss_dict =  self._predict_cbet(backbone_output)
            return predicted_action

    def _predict_cbet(self, 
        backbone_output, 
        action_seq: Optional[torch.Tensor] = None):
        backbone_output = einops.rearrange(backbone_output, "N T (A) -> (N) (T A)", A=self._act_dim)
        cbet_logits = self._map_to_cbet_preds_bin(backbone_output)
        cbet_offsets = self._map_to_cbet_preds_offset(backbone_output)
        # print(f"cbet_logits: {cbet_logits.shape}, cbet_offsets: {cbet_offsets.shape}")
        cbet_logits = einops.rearrange(
            cbet_logits, "(NT) (G C) -> (NT) G C", G=self._G
        ) #[N*T, G, C]
        cbet_offsets = einops.rearrange(
            cbet_offsets, "(NT) (G C WA) -> (NT) G C WA", G=self._G, C=self._C
        ) #[N*T, G, C, WA]
        cbet_probs = torch.softmax(cbet_logits, dim=-1)
        NT, G, choices = cbet_probs.shape
        sampled_centers = einops.rearrange(
            torch.multinomial(cbet_probs.view(-1, choices), num_samples=1),
            "(NT G) 1 -> NT G",
            NT=NT,
        )
        # print(f"cbet_logits: {cbet_logits.shape}, cebet_offsets: {cbet_offsets.shape}, cbet_probs: {cbet_probs.shape}")

        indices = (
            torch.arange(NT).unsqueeze(1).cuda(),
            torch.arange(self._G).unsqueeze(0).cuda(),
            sampled_centers,
        )
        # Use advanced indexing to sample the values
        sampled_offsets = cbet_offsets[indices]

        sampled_offsets = sampled_offsets.sum(dim=1) #N (T A)
        # print(f"sampled_offsets: {sampled_offsets.shape}, sampled_centers: {sampled_centers.shape}")
        centers = self.vqvae.draw_code_forward(sampled_centers).view(
            NT, -1, self._D
        )
        return_decoder_input = einops.rearrange(
            centers.clone().detach(), "NT G D -> NT (G D)"
        )
        decoded_action = (
            self.vqvae.get_action_from_latent(return_decoder_input)
            .clone()
            .detach()
        )  # NT, A
        sampled_offsets = einops.rearrange(
            sampled_offsets, "N (T A) -> N T A", T=self._act_seq_len, A=self._act_dim
        )
        predicted_action = decoded_action + sampled_offsets

        if action_seq is not None:
            # Figure out the loss for the actions.
            # First, we need to find the closest cluster center for each action.
            state_vq, action_bins = self.vqvae.get_code(
                action_seq
            )  # action_bins: N, G

            # print(f"state_vq: {state_vq.shape}, action_bins: {action_bins.shape}")

            # Now we can compute the loss.
            if action_seq.ndim == 2:
                action_seq = action_seq.unsqueeze(0)

            offset_loss = torch.nn.L1Loss()(action_seq, predicted_action)      
            
            # action_diff = F.mse_loss(
            #     einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
            #         :, -1, 0, :
            #     ],
            #     einops.rearrange(predicted_action, "(N T) W A -> N T W A", T=obs_w)[
            #         :, -1, 0, :
            #     ],
            # )  # batch, time, windowsize (t ... t+N), action dim -> [:, -1, 0, :] is for rollout
            action_diff_tot = F.mse_loss(action_seq, predicted_action) 
            # print(f"action_seq: {action_seq.shape}, decoded_action: {decoded_action.shape}")
            # action_diff_mean_res1 = (
            #     abs(
            #         einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
            #             :, -1, 0, :
            #         ]
            #         - einops.rearrange(decoded_action, "(N T) W A -> N T W A", T=obs_w)[
            #             :, -1, 0, :
            #         ]
            #     )
            # ).mean()
            # action_diff_mean_res2 = (
            #     abs(
            #         einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
            #             :, -1, 0, :
            #         ]
            #         - einops.rearrange(
            #             predicted_action, "(N T) W A -> N T W A", T=obs_w
            #         )[:, -1, 0, :]
            #     )
            # ).mean()
            # action_diff_max = (
            #     abs(
            #         einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
            #             :, -1, 0, :
            #         ]
            #         - einops.rearrange(
            #             predicted_action, "(N T) W A -> N T W A", T=obs_w
            #         )[:, -1, 0, :]
            #     )
            # ).max()

            cbet_loss1 = self._criterion(  # F.cross_entropy
                cbet_logits[:, 0, :],
                action_bins[:, 0],
            )
            cbet_loss2 = self._criterion(  # F.cross_entropy
                cbet_logits[:, 1, :],
                action_bins[:, 1],
            )
            cbet_loss = cbet_loss1 * 5 + cbet_loss2 * self._secondary_code_multiplier

            equal_total_code_rate = (
                torch.sum(
                    (
                        torch.sum((action_bins == sampled_centers).int(), axis=1) == G
                    ).int()
                )
                / NT
            )
            equal_single_code_rate = torch.sum(
                (action_bins[:, 0] == sampled_centers[:, 0]).int()
            ) / (NT)
            equal_single_code_rate2 = torch.sum(
                (action_bins[:, 1] == sampled_centers[:, 1]).int()
            ) / (NT)

            loss = cbet_loss + self._offset_loss_multiplier * offset_loss
            loss_dict = {
                "classification_loss": cbet_loss.detach().cpu().item(),
                "offset_loss": offset_loss.detach().cpu().item(),
                "total_loss": loss.detach().cpu().item(),
                "equal_total_code_rate": equal_total_code_rate,
                "equal_single_code_rate": equal_single_code_rate,
                "equal_single_code_rate2": equal_single_code_rate2,
                # "action_diff": action_diff.detach().cpu().item(),
                "action_diff_tot": action_diff_tot.detach().cpu().item(),
                # "action_diff_mean_res1": action_diff_mean_res1.detach().cpu().item(),
                # "action_diff_mean_res2": action_diff_mean_res2.detach().cpu().item(),
                # "action_diff_max": action_diff_max.detach().cpu().item(),
            }
            # print(f"loss_dict: {loss_dict}")
            return predicted_action, loss, loss_dict
        
        return predicted_action, None, {}
