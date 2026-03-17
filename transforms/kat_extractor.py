from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from environments.specs import CameraSpec, DataSpecs, DepthStream, PointMapStream
from transforms.base_transform import NormalizingTransform

log = logging.getLogger(__name__)


class KATExtractorTransform(NormalizingTransform, nn.Module):
    ref_features: torch.Tensor

    def __init__(
        self,
        specs: DataSpecs,
        pcd_keys: list[str],
        num_ref_features: float = 20,
        feature_dim: int = 768,
    ):
        super().__init__()

        self.pcd_keys = pcd_keys
        if isinstance(self.pcd_keys, str):
            self.pcd_keys = [self.pcd_keys]
        self.num_ref_features = num_ref_features
        self._specs = specs

        self.per_demonstration_features = {key: [] for key in self.pcd_keys}

        self.ref_features = torch.empty(
            (len(self.pcd_keys), self.num_ref_features, feature_dim)
        )

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def best_buddies(
        self,
        A: torch.Tensor,  # [Na, F], normalized
        B: torch.Tensor,  # [Nb, F], normalized
    ):
        """
        Compute Best-Buddies / mutual nearest-neighbor pairs between A and B.

        Assumes A and B are already L2-normalized, so cosine similarity is A @ B.T.

        Returns:
            pairs_a: LongTensor [M]   indices into A
            pairs_b: LongTensor [M]   indices into B
            pair_sims: Tensor [M]     cosine similarities of the mutual matches
            nn_b_for_a: LongTensor [Na]
            nn_a_for_b: LongTensor [Nb]
        """
        # cosine similarity matrix: [Na, Nb]
        S = A @ B.T

        # nearest neighbor in B for each point in A
        nn_b_for_a = S.argmax(dim=1)  # [Na]

        # nearest neighbor in A for each point in B
        nn_a_for_b = S.argmax(dim=0)  # [Nb]

        # mutual condition:
        # A[i] -> B[j] and B[j] -> A[i]
        idx_a = torch.arange(A.shape[0], device=A.device)
        mutual_mask = nn_a_for_b[nn_b_for_a] == idx_a  # [Na]

        pairs_a = idx_a[mutual_mask]  # [M]
        pairs_b = nn_b_for_a[mutual_mask]  # [M]
        pair_sims = S[pairs_a, pairs_b]  # [M]

        return pairs_a, pairs_b, pair_sims, nn_b_for_a, nn_a_for_b

    def compute_all_best_buddies(self, feature_maps: List[torch.Tensor]):
        """
        Precompute best-buddy matches for every ordered pair of demos.

        Returns:
            bb[(i, j)] = {
                "pairs_i": LongTensor [M],
                "pairs_j": LongTensor [M],
                "pair_sims": Tensor [M],
            }
        """
        bb = {}
        for i, A in enumerate(feature_maps):
            for j, B in enumerate(feature_maps):
                if i == j:
                    continue
                pairs_i, pairs_j, pair_sims, _, _ = self.best_buddies(A, B)
                bb[(i, j)] = {
                    "pairs_i": pairs_i,
                    "pairs_j": pairs_j,
                    "pair_sims": pair_sims,
                }
        return bb

    def calculate_ref_features(self, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        device = feature_maps[0].device
        best_buddies = self.compute_all_best_buddies(feature_maps)

        candidates = []

        for i, Xi in enumerate(feature_maps):
            Ni = Xi.shape[0]
            # How many mutual matches does each ref featureset have in other demonstrations?
            coverage = torch.zeros(Ni, device=device, dtype=torch.long)
            # What is the average similarity of those matches?
            sim_sum = torch.zeros(Ni, device=device)

            for j in range(len(feature_maps)):
                if i == j:
                    continue

                pairs_i = best_buddies[(i, j)]["pairs_i"]
                pair_sims = best_buddies[(i, j)]["pair_sims"]

                if pairs_i.numel() == 0:
                    continue

                coverage[pairs_i] += 1
                sim_sum[pairs_i] += pair_sims

            avg_sim = sim_sum / torch.clamp(coverage, min=1.0)

            for p in range(Ni):
                cov = int(coverage[p].item())

                candidates.append(
                    {
                        "desc": Xi[p],
                        "score": 1000.0 * cov + avg_sim[p].item(),
                    }
                )

        candidates.sort(key=lambda x: x["score"], reverse=True)

        selected_desc = [cand["desc"] for cand in candidates[: self.num_ref_features]]

        ref_desc = torch.stack(selected_desc, dim=0)
        return ref_desc

    def calculate_ref_features_all_cams(self):
        for i, key in enumerate(self.pcd_keys):
            features = self.per_demonstration_features[key]
            ref_features = self.calculate_ref_features(features)
            self.ref_features[i] = ref_features

    def __getstate__(self):
        if self.per_demonstration_features is not None:
            self.calculate_ref_features_all_cams()

        state = self.__dict__.copy()
        state["per_demonstration_features"] = None  # do not pickle the model
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def call_trajectory(self, tensordict):
        for pcd_key in self.pcd_keys:
            pcd = tensordict["obs", pcd_key]
            self.per_demonstration_features[pcd_key].append(pcd["features"][0])

        return tensordict

    def match_reference_features_to_pointcloud(self, ref_desc: torch.Tensor, pcd: dict):
        points = pcd["points"]
        if points.ndim == 4:
            points = points[..., 0, :]

        B = points.shape[0]
        device = points.device

        feats = pcd["features"]
        device = feats.device
        ref_desc = ref_desc.to(device)

        point_offsets = points.offsets()
        feat_offsets = feats.offsets()

        batch_sizes = feat_offsets[1:] - feat_offsets[:-1]
        if torch.any(batch_sizes == 0):
            pass
            # raise RuntimeError(
            #     f"Encountered empty point cloud(s): batch_sizes={batch_sizes}"
            # )

        sim_values = feats.values() @ ref_desc.T
        sim = torch.nested.nested_tensor_from_jagged(sim_values, offsets=feat_offsets)

        if sim.values().numel() == 0:
            # No features in this point cloud, return zeros
            return_dict = {
                "points": torch.zeros((B, self.num_ref_features, 3), device=device),
                "features": torch.zeros(
                    (B, self.num_ref_features, feats.shape[-1]), device=device
                ),
            }
            if "colors" in pcd:
                return_dict["colors"] = torch.zeros(
                    (B, self.num_ref_features, 3), device=device
                )
            return return_dict

        # for each batch item and each reference feature, choose best point
        indices = sim.argmax(dim=1)

        cum_indices = indices + feat_offsets[:-1].unsqueeze(-1)

        # Setting empty batch items to zero index (which will be ignored later) to avoid out-of-bounds indexing
        cum_indices[batch_sizes == 0, :] = 0
        if torch.any(indices == -1) or torch.any(
            cum_indices >= feats.values().shape[0]
        ):
            raise RuntimeError(
                f"Invalid indices found: indices={indices}, cum_indices={cum_indices}, feat_offsets={feat_offsets}, feats_values_shape={feats.values().shape}"
            )

        return_dict = {
            "points": points.values()[
                cum_indices
            ],  # torch.zeros((B, self.num_ref_features, 3), device=device),
            "features": feats.values()[
                cum_indices
            ],  # torch.zeros((B, self.num_ref_features, feats.shape[-1]), device=device),
        }

        return_dict["points"][batch_sizes == 0, :] = 0.0
        return_dict["features"][batch_sizes == 0, :] = 0.0

        if "colors" in pcd:
            colors = pcd["colors"]
            return_dict["colors"] = colors.values()[cum_indices]
            return_dict["colors"][batch_sizes == 0, :] = 0.0

        return return_dict

    def forward(self, tensordict):
        for i, key in enumerate(self.pcd_keys):
            tensordict["obs", key] = self.match_reference_features_to_pointcloud(
                self.ref_features[i], tensordict["obs", key]
            )

        return tensordict

    def reverse(self, tensordict):
        return tensordict
