from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class DMRResult:
    text: Tensor
    visual: Tensor
    raw_visual: Tensor
    visual_reliability: Tensor
    image_score: Tensor
    loss: Tensor
    losses: dict[str, Tensor]


class DMR(nn.Module):
    """Disentangled modality reconstruction core.

    The architecture follows the paper's public method description. Loss weights and
    exact reductions remain audit-pending until the full paper/supplementary equations
    are available; callers must not label this exact reproduction yet.
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 256,
        alpha: float = 0.005,
        beta: float = 0.01,
        ablation: str | None = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ablation = ablation
        self.text_shared = nn.Linear(input_dim, latent_dim)
        self.text_specific = nn.Linear(input_dim, latent_dim)
        self.visual_shared = nn.Linear(input_dim, latent_dim)
        self.visual_specific = nn.Linear(input_dim, latent_dim)
        self.text_logvar = nn.Linear(input_dim, latent_dim)
        self.visual_logvar = nn.Linear(input_dim, latent_dim)
        self.text_decoder = nn.Linear(latent_dim * 2, input_dim)
        self.visual_decoder = nn.Linear(latent_dim * 2, input_dim)

    @staticmethod
    def _sample(mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.exp(0.5 * logvar).clamp_max(100.0) * torch.randn_like(mu)

    def forward(self, text: Tensor, visual: Tensor) -> DMRResult:
        ts = self.text_shared(text)
        vs = self.visual_shared(visual)
        tm = self.text_specific(text)
        vm = self.visual_specific(visual)
        tz = self._sample(ts, self.text_logvar(text))
        vz = self._sample(vs, self.visual_logvar(visual))
        text_recon = self.text_decoder(torch.cat([tz, tm], -1))
        visual_recon = self.visual_decoder(torch.cat([vz, vm], -1))
        cross_text_recon = self.text_decoder(torch.cat([vz, tm], -1))
        cross_visual_recon = self.visual_decoder(torch.cat([tz, vm], -1))
        recon = (text_recon - text).pow(2).mean() + (visual_recon - visual).pow(2).mean()
        cross_reconstruction = (cross_text_recon - text).pow(2).mean() + (cross_visual_recon - visual).pow(2).mean()
        alignment = (ts - vs).pow(2).mean()
        intra_disentangle = (tm * ts.detach()).pow(2).mean() + (vm * vs.detach()).pow(2).mean()
        inter_disentangle = (tm * vm).pow(2).mean()
        image_score = (visual_recon - visual).pow(2).mean(-1)
        reliability = torch.exp(-image_score).clamp(0.0, 1.0)
        raw_visual = vs + vm
        visual_output = raw_visual if self.ablation == "w/o C_M" else raw_visual * reliability.unsqueeze(-1)
        terms = {
            "reconstruction": recon,
            "cross_reconstruction": cross_reconstruction,
            "alignment": alignment,
            "intra_disentanglement": intra_disentangle,
            "inter_disentanglement": inter_disentangle,
        }
        loss = recon
        if self.ablation != "w/o z_tI":
            loss = loss + cross_reconstruction
        if self.ablation != "w/o L_DA":
            loss = loss + alignment
        if self.ablation != "w/o L_IntraDD":
            loss = loss + self.alpha * intra_disentangle
        if self.ablation != "w/o L_InterDD":
            loss = loss + self.beta * inter_disentangle
        return DMRResult((ts + tm), visual_output, raw_visual, reliability, image_score, loss, terms)


class MTC(nn.Module):
    """Multi-grained triplet confidence: local, graph-structure and path scores."""

    def __init__(
        self,
        dim: int = 256,
        relation_dim: int = 256,
        lambda_structure: float = 0.5,
        delta_path: float = 1.5,
        ablation: str | None = None,
    ) -> None:
        super().__init__()
        self.lambda_structure = lambda_structure
        self.delta_path = delta_path
        self.ablation = ablation
        self.relation = nn.Embedding(4096, relation_dim)
        self.local = nn.Sequential(nn.Linear(dim * 3, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.global_structure = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.global_path = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, head: Tensor, relation_id: Tensor, tail: Tensor, neighbor: Tensor, path: Tensor) -> dict[str, Tensor]:
        r = self.relation(relation_id)
        local = self.local(torch.cat([head, r, tail], -1)).squeeze(-1)
        structure = self.global_structure(torch.cat([head, neighbor], -1)).squeeze(-1)
        path_score = self.global_path(torch.cat([head, path], -1)).squeeze(-1)
        local_weight = 0.0 if self.ablation == "w/o S_LT" else 1.0
        structure_weight = 0.0 if self.ablation == "w/o S_GS" else self.lambda_structure
        path_weight = 0.0 if self.ablation == "w/o S_GP" else self.delta_path
        total = local_weight * local + structure_weight * structure + path_weight * path_score
        return {"local": local, "global_structure": structure, "global_path": path_score, "total": total}

    @staticmethod
    def ranking_loss(
        positive: Tensor,
        negative: Tensor,
        margin: float = 1.0,
        confidence: Tensor | None = None,
    ) -> Tensor:
        # Scores are suspiciousness distances: clean triplets should be lower
        # than generated corruptions, matching the paper's descending error rank.
        per_item = torch.relu(margin + positive - negative)
        if confidence is not None:
            per_item = per_item * confidence
        return per_item.mean()

