from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn
from tensordict import NonTensorData, TensorDict
from torch import Tensor
from torch_geometric.data import Batch, Data

from environments.specs import DataSpecs, EmbedSpec, PointCloudSpec
from transforms.base_transform import KeyMapping, Transform
from utils.nested import (
    cat_nested,
    make_jagged_nested_tensors_compatible,
    pyg_to_nested_tensor,
    to_strided_tensor,
)

log = logging.getLogger(__name__)


class PaktTokenizer(Transform, nn.Module):

    def __init__(
        self,
        specs: DataSpecs,
        embed_dim: int,
        feature_encoder: Callable[[int], nn.Linear],
        color_encoder: Callable[[int], nn.Linear],
        pos_encoder: Callable[[int, int], nn.Linear],
    ):
        super().__init__()

        self._input_spec = specs.obs

        self.feature_encoder = feature_encoder()
        self.color_encoder = color_encoder()
        self.pos_encoder = pos_encoder()

        # assert (
        #     feature_encoder.out_features
        #     == color_encoder.out_features
        #     == pos_encoder.out_features
        #     == embed_dim
        # )

        self.gripper_points_id_embedding = nn.Embedding(5, embed_dim)
        self.token_type_embedding = nn.Embedding(3, embed_dim)  # target, tool, gripper
        self.timestep_embedding = nn.Embedding(15, embed_dim)

        obs_embed_spec = EmbedSpec(
            embed_dim=embed_dim,
            n_tokens=None,
            fixed_shape=False,
        )
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        obs_specs["embed"] = obs_embed_spec

        self._output_specs = specs.replace(
            obs=obs_specs,
        )
        self.embed_dim = embed_dim

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __tokenize_pointcloud(
        self,
        point_pos: Tensor,
        point_color: Tensor,
        point_features: Tensor,
        token_type: int,
    ) -> Tensor:
        """
        Tokenize a point cloud into a set of embedding tokens.

        Args:
            point_pos (Tensor): Point positions, shape (B, N, 3).
            point_color (Tensor): Point colors, shape (B, N, 3).
            point_features (Tensor): Point features, shape (B, N, F).
            token_type (int): Token type identifier (0: target, 1: tool, 2: gripper).

        Returns:
            Tensor: Point cloud tokens, shape (B, N, D).
        """
        # Encode position
        pos_embed = self.pos_encoder(point_pos)

        # TODO: Make color encoder optional
        # Encode color
        color_embed = self.color_encoder(point_color)

        # Encode features
        feature_embed = self.feature_encoder(point_features)

        # Add token type embedding
        token_type_embed = self.token_type_embedding(
            torch.tensor([token_type], device=point_pos.device)
        )

        # Sum embeddings
        token_embed = pos_embed + color_embed + feature_embed + token_type_embed

        return token_embed

    def __tokenize_gripper_points(
        self,
        point_pos: Tensor,
        gripper_point_ids: Tensor,
    ) -> Tensor:
        """
        Tokenize gripper points into a set of embedding tokens.

        Args:
            point_pos (Tensor): Gripper point positions, shape (B, N, 3).
            gripper_point_ids (Tensor): Gripper point IDs, shape (B, N).

        Returns:
            Tensor: Gripper point tokens, shape (B, N, D).
        """
        B, N, _ = point_pos.shape

        # Encode position
        pos_embed = self.pos_encoder(point_pos)

        # Add gripper point ID embedding
        gripper_id_embed = self.gripper_points_id_embedding(gripper_point_ids)

        # Add token type embedding for gripper points (token_type=2)
        token_type_embed = self.token_type_embedding(
            torch.tensor([2], device=point_pos.device)
        )

        # Sum embeddings
        token_embed = pos_embed + gripper_id_embed + token_type_embed

        return token_embed

    def __tokenize_gripper_action_points(
        self,
        point_pos: Tensor,
        gripper_point_ids: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """
        Tokenize gripper action points into a set of embedding tokens.

        Args:
            point_pos (Tensor): Gripper action point positions, shape (B, T, N, 3).
            gripper_point_ids (Tensor): Gripper point IDs, shape (B, T, N).
            timesteps (Tensor): Timesteps, shape (B, T).

        Returns:
            Tensor: Gripper action point tokens, shape (B, T, N, D).
        """
        B, N, T, _ = point_pos.shape

        # Encode position
        pos_embed = self.pos_encoder(point_pos)

        # Add gripper point ID embedding
        gripper_id_embed = self.gripper_points_id_embedding(gripper_point_ids)

        # Add timestep embedding
        timestep_embed = self.timestep_embedding(timesteps)

        # Add token type embedding for gripper points (token_type=2)
        token_type_embed = self.token_type_embedding(
            torch.tensor([2], device=point_pos.device)
        )

        # Sum embeddings
        token_embed = (
            pos_embed
            + gripper_id_embed.unsqueeze(1)
            + timestep_embed.unsqueeze(0)
            + token_type_embed.unsqueeze(0)
        )

        return token_embed

    def __tokenize_tool_action_points(
        self,
        point_pos: Tensor,
        point_color: Tensor,
        point_features: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """
        Tokenize tool action points into a set of embedding tokens.

        Args:
            point_pos (Tensor): Tool action point positions, shape (B, T, N, 3).
            point_color (Tensor): Tool action point colors, shape (B, T, N, 3).
            point_features (Tensor): Tool action point features, shape (B, T, N, F).
            timesteps (Tensor): Timesteps, shape (B, T).

        Returns:
            Tensor: Tool action point tokens, shape (B, T, N, D).
        """
        B, T, N, _ = point_pos.shape

        # Encode position
        pos_embed = self.pos_encoder(point_pos)

        # Encode color
        color_embed = self.color_encoder(point_color)
        color_embed = make_jagged_nested_tensors_compatible(pos_embed, color_embed)

        # Encode features
        feature_embed = self.feature_encoder(point_features)
        feature_embed = make_jagged_nested_tensors_compatible(pos_embed, feature_embed)

        # Add timestep embedding
        timestep_embed = self.timestep_embedding(timesteps)

        # Add token type embedding for tool points (token_type=1)
        token_type_embed = self.token_type_embedding(
            torch.tensor([1], device=point_pos.device)
        )

        # Sum embeddings
        token_embed = (
            pos_embed
            + color_embed.unsqueeze(-2)
            + feature_embed.unsqueeze(-2)
            + timestep_embed.unsqueeze(0)
            + token_type_embed.unsqueeze(0)
        )

        return token_embed

    def forward(self, batch: TensorDict) -> TensorDict:
        obs = batch["obs"]
        B = batch.batch_size[0]
        actions = batch["noisy_action"]  # (B, T, N_a, 3)
        device = actions.device

        pyg_to_nested_tensor(
            obs["target_points"].pos,
            batch=obs["target_points"].batch,
            ptr=obs["target_points"].ptr,
        )  # ensure contiguous

        # target point tokens
        target_points = obs["target_points"]
        target_points_pos = pyg_to_nested_tensor(
            target_points.pos,
            batch=target_points.batch,
            ptr=target_points.ptr,
        )  # (B, N_t, 3)
        target_points_color = pyg_to_nested_tensor(
            target_points.color,
            batch=target_points.batch,
            ptr=target_points.ptr,
        )  # (B, N_t, 3)
        target_points_features = pyg_to_nested_tensor(
            target_points.x,
            batch=target_points.batch,
            ptr=target_points.ptr,
        )  # (B, N_t, F)

        target_points_tokens = self.__tokenize_pointcloud(
            point_pos=target_points_pos,
            point_color=target_points_color,
            point_features=target_points_features,
            token_type=0,  # target
        )  # (B, N_t, D)

        # tool point tokens
        tool_points = obs["tool_points"]
        tool_points_pos = pyg_to_nested_tensor(
            tool_points.pos,
            batch=tool_points.batch,
            ptr=tool_points.ptr,
        )  # (B, N_tool, 3)
        tool_points_color = pyg_to_nested_tensor(
            tool_points.color,
            batch=tool_points.batch,
            ptr=tool_points.ptr,
        )  # (B, N_tool, 3)
        tool_points_features = pyg_to_nested_tensor(
            tool_points.x,
            batch=tool_points.batch,
            ptr=tool_points.ptr,
        )  # (B, N_tool, F)
        tool_points_tokens = self.__tokenize_pointcloud(
            point_pos=tool_points_pos,
            point_color=tool_points_color,
            point_features=tool_points_features,
            token_type=1,  # tool
        )  # (B, N_tool, D)

        # gripper point tokens
        gripper_points = obs["gripper_points"]
        gripper_points_pos = to_strided_tensor(
            pyg_to_nested_tensor(
                gripper_points.pos,
                batch=gripper_points.batch,
                ptr=gripper_points.ptr,
            )
        )  # (B, N_g, 3)
        gripper_points_ids = torch.arange(gripper_points_pos.shape[1], device=device)
        # (B, N_g) # should be 5

        gripper_points_tokens = self.__tokenize_gripper_points(
            point_pos=gripper_points_pos,
            gripper_point_ids=gripper_points_ids,
        )  # (B, N_g, D)

        action_points_pos = actions  # (B, T, N_a, 3)

        # actions point tokens
        gripper_action_pos = torch.stack([x[:5] for x in action_points_pos.unbind(0)])
        # (B, T, 5) # first 5 dims are robot actions
        gripper_action_ids = torch.arange(5, device=device)
        # (B, T, 5)

        gripper_action_timesteps = torch.arange(
            gripper_action_pos.shape[2], device=device
        )  # (B, T)
        gripper_action_tokens = self.__tokenize_gripper_action_points(
            point_pos=gripper_action_pos,
            gripper_point_ids=gripper_action_ids,
            timesteps=gripper_action_timesteps,
        )  # (B, T, 5, D)

        tool_action_pos = torch.nested.nested_tensor(
            [x[5:] for x in action_points_pos.unbind(0)], layout=torch.jagged
        )  # (B, T, N_tool, 3) # rest are tool actions
        tool_action_features = tool_points_features  # (B, T, N_tool, F)
        tool_action_color = tool_points_color  # (B, T, N_tool, 3)
        tool_action_timesteps = torch.arange(
            tool_action_pos.shape[2], device=device
        )  # (B, T)
        tool_action_tokens = self.__tokenize_tool_action_points(
            point_pos=tool_action_pos,
            point_color=tool_action_color,
            point_features=tool_action_features,
            timesteps=tool_action_timesteps,
        )  # (B, T, N_tool, D)

        # concatenate along N dimension of embedding
        # obs_embed: (B, N, D)
        obs_tokens = cat_nested(
            [target_points_tokens, tool_points_tokens, gripper_points_tokens], dim=1
        )
        action_tokens = cat_nested([gripper_action_tokens, tool_action_tokens], dim=1)

        action_tokens = torch.nested.nested_tensor_from_jagged(
            values=action_tokens.values().view(-1, self.embed_dim),
            offsets=action_tokens.offsets() * 15,
        )

        batch["obs"]["embed"] = obs_tokens
        # batch["action_embed"] = action_tokens
        return batch, action_tokens
