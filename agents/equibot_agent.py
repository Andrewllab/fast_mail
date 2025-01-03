import logging
import stat

from attr import has
import einops
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from agents.base_agent import BaseAgent
from agents.models.equibot.utils.diffusion import lr_scheduler
from agents.models.equibot.utils.misc import to_torch
from agents.models.equibot.utils.norm import Normalizer
from agents.models.equibot.utils.diffusion.lr_scheduler import get_scheduler

log = logging.getLogger(__name__)


class EquiBotAgent(BaseAgent):
    def __init__(
        self,
        model: DictConfig,
        obs_encoders: DictConfig,
        language_encoders: DictConfig,
        device: str,
        state_dim: int,
        latent_dim: int,
        optimization: DictConfig,
        obs_seq_len: int,
        act_seq_len: int,
    ):
        super().__init__(
            model=model,
            obs_encoders=obs_encoders,
            language_encoders=language_encoders,
            device=device,
            state_dim=state_dim,
            latent_dim=latent_dim,
            obs_seq_len=obs_seq_len,
            act_seq_len=act_seq_len,
        )

        self.pc_normalizer = None
        self.state_normalizer = None
        self.ac_normalizer = None

        self.pc_scale = None

        self.optimizer_config = optimization

        self.dof = self.model.dof
        self.num_eef = self.model.num_eef

    def configure_optimizers(self, num_training_steps: int):
        optimizer = hydra.utils.instantiate(
            self.optimizer_config, params=self.parameters()
        )

        # lr_scheduler = get_scheduler(
        #     name="cosine",
        #     optimizer=optimizer,
        #     num_warmup_steps=500,
        #     num_training_steps=num_training_steps,
        # )
        lr_scheduler = None

        return optimizer, lr_scheduler

    def _init_normalizers(self, obs_dict, action):
        if self.ac_normalizer is None:
            flattened_gt_action = action.view(-1, self.dof)
            if self.dof == 7:
                indices = [[0], [1, 2, 3], [4, 5, 6]]
            elif self.dof == 4:
                indices = [[0], [1, 2, 3]]
            else:
                indices = None
            ac_normalizer = Normalizer(
                flattened_gt_action, symmetric=True, indices=indices
            )
            self.ac_normalizer = Normalizer(
                {
                    "min": ac_normalizer.stats["min"].tile((self.num_eef,)),
                    "max": ac_normalizer.stats["max"].tile((self.num_eef,)),
                }
            )
            print(f"Action normalization stats: {self.ac_normalizer.stats}")
        if self.state_normalizer is None:
            # dof layout: maybe gripper open/close, xyz, maybe rot
            if self.dof == 3:
                self.state_normalizer = ac_normalizer
            else:
                self.state_normalizer = Normalizer(
                    {
                        "min": ac_normalizer.stats["min"][1:4],
                        "max": ac_normalizer.stats["max"][1:4],
                    }
                )
            self.model.state_normalizer = self.state_normalizer
        if self.pc_normalizer is None:
            self.pc_normalizer = self.state_normalizer
            self.model.pc_normalizer = self.pc_normalizer

        # compute action scale relative to point cloud scale
        pc = obs_dict["pc"].reshape(-1, obs_dict["pc"].shape[-2], 3)
        centroid = pc.mean(1, keepdim=True)
        centered_pc = pc - centroid
        pc_scale = centered_pc.norm(dim=-1).mean()
        ac_scale = ac_normalizer.stats["max"].max()
        self.pc_scale = pc_scale / ac_scale
        self.model.pc_scale = self.pc_scale

    def act(self, obs, return_dict=False, debug=False):
        self.train(False)
        # assert isinstance(obs["pc"][0][0], torch.Tensor)
        # if len(obs["robot_states"].shape) == 3:
        #     assert len(obs["pc"][0].shape) == 2  # (obs_horizon, N, 3)
        #     obs["pc"] = [[x] for x in obs["pc"]]
        #     for k in obs:
        #         if k != "pc" and isinstance(obs[k], torch.Tensor):
        #             obs[k] = obs[k][:, None]
        #     has_batch_dim = False
        # elif len(obs["robot_states"].shape) == 4:
        #     assert len(obs["pc"][0][0].shape) == 2  # (obs_horizon, B, N, 3)
        #     has_batch_dim = True
        # else:
        #     raise ValueError("Input format not recognized.")

        has_batch_dim = True
        ac_dim = self.num_eef * self.dof
        batch_size = obs["pc"].shape[0]
        state = obs["robot_states"]
        
        # state = obs["robot_states"].reshape(tuple(obs["robot_states"].shape[:2]) + (-1,))

        # process the point clouds
        # some point clouds might be invalid
        # if this occurs, exclude these batch items
        # xyzs = []
        ac = torch.zeros([batch_size, self.model.pred_horizon, ac_dim]).to(self.device)
        if return_dict:
            ac_dict = []
            for i in range(batch_size):
                ac_dict.append(None)
        forward_idxs = list(torch.arange(batch_size).to(self.device))
        # for pcs in obs["pc"]:
        #     xyzs.append([])
        #     for batch_idx, xyz in enumerate(pcs):
        #         if not batch_idx in forward_idxs:
        #             xyzs[-1].append(torch.zeros((self.num_points, 3)))
        #         elif xyz.shape[0] == 0:
        #             # no points in point cloud, return no-op action
        #             forward_idxs.remove(batch_idx)
        #             xyzs[-1].append(torch.zeros((self.num_points, 3)))
        #         elif self.shuffle_pc:
        #             choice = torch.from_numpy(np.random.choice(
        #                 xyz.shape[0], self.num_points, replace=True
        #             ))
        #             xyz = xyz[choice, :]
        #             xyzs[-1].append(xyz)
        #         else:
        #             step = xyz.shape[0] // self.num_points
        #             xyz = xyz[::step, :][: self.num_points]
        #             xyzs[-1].append(xyz)

        # if len(forward_idxs) > 0:
        #     torch_obs = dict(
        #         pc=torch.tensor(xyzs).swapaxes(0, 1)[forward_idxs],
        #         state=state.swapaxes(0, 1)[forward_idxs]
        #     )
        #     for k in obs:
        #         if not k in ["pc", "state"] and isinstance(obs[k], torch.Tensor):
        #             torch_obs[k] = (
        #                 obs[k].swapaxes(0, 1)[forward_idxs]
        #             )
        # else:
        #     raw_ac_dict = torch.zeros(
        #         (batch_size, self.actor.pred_horizon, self.actor.action_dim)
        #     ).to(self.actor.device)

        torch_obs = dict(pc=obs["pc"], state=state)
        raw_ac_dict = self.model(torch_obs, debug=debug)

        for i, idx in enumerate(forward_idxs):
            if return_dict:
                ac_dict[idx] = {k: v[i] for k, v in raw_ac_dict.items()}
            unnormed_action = self.ac_normalizer.unnormalize(raw_ac_dict["ac"][i])
            ac[idx] = unnormed_action

        if not has_batch_dim:
            ac = ac[0]
            if return_dict:
                ac_dict = ac_dict[0]
        if return_dict:
            return ac, ac_dict
        else:
            return ac

    def forward(self, obs_dict, action=None):
        if action is None:
            return self.act(obs_dict)

        self.train()

        pc = obs_dict["pc"]
        # rgb = batch["rgb"]
        state = obs_dict["robot_states"]
        gt_action = action  # torch.Size([32, 16, self.num_eef * self.dof])

        if self.pc_scale is None:
            self._init_normalizers(obs_dict, action)
        pc = self.pc_normalizer.normalize(pc)
        gt_action = self.ac_normalizer.normalize(gt_action)

        pc_shape = pc.shape
        batch_size = B = pc.shape[0]
        Ho = self.model.obs_horizon
        Hp = self.model.pred_horizon

        if self.model.obs_mode == "state":
            z_pos, z_dir, z_scalar = self.model._convert_state_to_vec(state)
            z_pos = self.state_normalizer.normalize(z_pos)
            if self.dof > 4:
                z = torch.cat([z_pos, z_dir], dim=-2)
            else:
                z = z_pos
        else:
            feat_dict = self.model.encoder_handle(pc, target_norm=self.pc_scale)

            center = (
                feat_dict["center"].reshape(B, Ho, 1, 3)[:, [-1]].repeat(1, Ho, 1, 1)
            )
            scale = feat_dict["scale"].reshape(B, Ho, 1, 1)[:, [-1]].repeat(1, Ho, 1, 1)
            z_pos, z_dir, z_scalar = self.model._convert_state_to_vec(state)
            z_pos = self.state_normalizer.normalize(z_pos)
            z_pos = (z_pos - center) / scale
            z = feat_dict["so3"]
            z = z.reshape(B, Ho, -1, 3)
            if self.dof > 4:
                z = torch.cat([z, z_pos, z_dir], dim=-2)
            else:
                z = torch.cat([z, z_pos], dim=-2)
        obs_cond_vec, obs_cond_scalar = z.reshape(B, -1, 3), (
            z_scalar.reshape(B, -1) if z_scalar is not None else None
        )

        if self.model.obs_mode.startswith("pc"):
            if self.model.ac_mode == "abs":
                center = (
                    feat_dict["center"]
                    .reshape(B, Ho, 1, 3)[:, [-1]]
                    .repeat(1, Hp, 1, 1)
                )
            else:
                center = 0
            scale = feat_dict["scale"].reshape(B, Ho, 1, 1)[:, [-1]].repeat(1, Hp, 1, 1)
            gt_action = gt_action.reshape(B, Hp, self.num_eef, self.dof)
            if self.dof == 4:
                gt_action = torch.cat(
                    [gt_action[..., :1], (gt_action[..., 1:] - center) / scale], dim=-1
                )
                gt_action = gt_action.reshape(B, Hp, -1)
            elif self.dof == 3:
                gt_action = (gt_action - center) / scale
            elif self.dof == 7:
                gt_action = torch.cat(
                    [
                        gt_action[..., :1],
                        (gt_action[..., 1:4] - center) / scale,
                        gt_action[..., 4:],
                    ],
                    dim=-1,
                )
                gt_action = gt_action.reshape(B, Hp, -1)
            else:
                raise ValueError(f"Dof {self.dof} not supported.")
        vec_eef_action, vec_gripper_action = self.model._convert_action_to_vec(
            gt_action
        )
        if self.dof != 7:
            noise = torch.randn(gt_action.shape, device=self.device)
            vec_eef_noise, vec_gripper_noise = self.model._convert_action_to_vec(
                noise
            )  # to debug
        else:
            vec_eef_noise = torch.randn_like(vec_eef_action, device=self.device)
            vec_gripper_noise = torch.randn_like(vec_gripper_action, device=self.device)

        timesteps = torch.randint(
            0,
            self.model.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=self.device,
        ).long()

        if vec_gripper_action is not None:
            noisy_eef_actions = self.model.noise_scheduler.add_noise(
                vec_eef_action, vec_eef_noise, timesteps
            )
            noisy_gripper_actions = self.model.noise_scheduler.add_noise(
                vec_gripper_action, vec_gripper_noise, timesteps
            )

            vec_eef_noise_pred, vec_gripper_noise_pred = (
                self.model.noise_pred_net_handle(
                    noisy_eef_actions.permute(0, 3, 1, 2),
                    timesteps,
                    scalar_sample=noisy_gripper_actions.permute(0, 2, 1),
                    cond=obs_cond_vec,
                    scalar_cond=obs_cond_scalar,
                )
            )
            vec_eef_noise_pred = vec_eef_noise_pred.permute(0, 2, 3, 1)
            vec_gripper_noise_pred = vec_gripper_noise_pred.permute(0, 2, 1)
            if self.dof != 7:
                noise_pred = self.model._convert_action_to_scalar(
                    vec_eef_noise_pred, vec_gripper_noise_pred
                ).view(noise.shape)
        else:
            noisy_eef_actions = self.model.noise_scheduler.add_noise(
                vec_eef_action, vec_eef_noise, timesteps
            )

            vec_noise_pred = self.model.noise_pred_net_handle(
                noisy_eef_actions.permute(0, 3, 1, 2),
                timesteps,
                cond=obs_cond_vec,
                scalar_cond=obs_cond_scalar,
            )[0].permute(0, 2, 3, 1)
            if self.dof != 7:
                noise_pred = self.model._convert_action_to_scalar(vec_noise_pred).view(
                    noise.shape
                )

        if self.dof == 7:
            n_vec = np.prod(vec_eef_noise_pred.shape)
            n_sca = np.prod(vec_gripper_noise_pred.shape)
            k = (n_vec) / (n_vec + n_sca)
            loss = nn.functional.mse_loss(
                vec_eef_noise_pred, vec_eef_noise
            ) * k + nn.functional.mse_loss(
                vec_gripper_noise_pred, vec_gripper_noise
            ) * (
                1 - k
            )
        else:
            loss = nn.functional.mse_loss(noise_pred, noise)
        if torch.isnan(loss):
            print(f"Loss is nan, please investigate.")
            import pdb

            pdb.set_trace()

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # self.lr_scheduler.step()

        self.model.step_ema()

        metrics = {
            "loss": loss,
            "normalized_gt_ac_max": np.max(
                np.abs(vec_eef_action.reshape(-1, 3).detach().cpu().numpy()), axis=0
            ).mean(),
        }
        if self.dof == 7:
            metrics.update(
                {
                    "mean_gt_eef_noise_norm": np.linalg.norm(
                        vec_eef_noise.detach().cpu().numpy(), axis=1
                    ).mean(),
                    "mean_pred_eef_noise_norm": np.linalg.norm(
                        vec_eef_noise_pred.detach().cpu().numpy(), axis=1
                    ).mean(),
                    "mean_gt_gripper_noise_norm": np.linalg.norm(
                        vec_gripper_noise.detach().cpu().numpy(), axis=1
                    ).mean(),
                    "mean_pred_gripper_noise_norm": np.linalg.norm(
                        vec_gripper_noise_pred.detach().cpu().numpy(), axis=1
                    ).mean(),
                }
            )
        else:
            metrics.update(
                {
                    "mean_gt_noise_norm": np.linalg.norm(
                        noise.reshape(gt_action.shape[0], -1).detach().cpu().numpy(),
                        axis=1,
                    ).mean(),
                    "mean_pred_noise_norm": np.linalg.norm(
                        noise_pred.reshape(gt_action.shape[0], -1)
                        .detach()
                        .cpu()
                        .numpy(),
                        axis=1,
                    ).mean(),
                }
            )

        return metrics

    def save_snapshot(self, save_path):
        state_dict = dict(
            actor=self.model.state_dict(),
            ema_model=self.model.ema.averaged_model.state_dict(),
            pc_scale=self.pc_scale,
            pc_normalizer=self.pc_normalizer.state_dict(),
            state_normalizer=self.state_normalizer.state_dict(),
            ac_normalizer=self.ac_normalizer.state_dict(),
        )
        torch.save(state_dict, save_path)

    def fix_checkpoint_keys(self, state_dict):
        fixed_state_dict = dict()
        for k, v in state_dict.items():
            if "encoder.encoder" in k:
                fixed_k = k.replace("encoder.encoder", "encoder")
            else:
                fixed_k = k
            if "handle" in k:
                continue
            fixed_state_dict[fixed_k] = v
        return fixed_state_dict

    def load_snapshot(self, save_path):
        state_dict = torch.load(save_path)
        self.state_normalizer = Normalizer(state_dict["state_normalizer"])
        self.model.state_normalizer = self.state_normalizer
        self.ac_normalizer = Normalizer(state_dict["ac_normalizer"])
        if self.model.obs_mode.startswith("pc"):
            self.pc_normalizer = self.state_normalizer
            self.model.pc_normalizer = self.pc_normalizer
        del self.model.encoder_handle
        del self.model.noise_pred_net_handle
        self.model.load_state_dict(self.fix_checkpoint_keys(state_dict["actor"]))
        self.model._init_torch_compile()
        self.model.ema.averaged_model.load_state_dict(
            self.fix_checkpoint_keys(state_dict["ema_model"])
        )
        self.pc_scale = state_dict["pc_scale"]
        self.model.pc_scale = self.pc_scale
