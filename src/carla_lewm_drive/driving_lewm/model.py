from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


VIT_CONFIGS = {
    "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
    "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
    "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
}


@dataclass(frozen=True)
class DrivingLeWMConfig:
    encoder_scale: str = "tiny"
    patch_size: int = 14
    image_size: int = 224
    action_dim: int = 15
    embed_dim: int = 192
    history_size: int = 3
    predictor_depth: int = 6
    predictor_heads: int = 6
    predictor_mlp_dim: int = 768
    dropout: float = 0.1
    sigreg_weight: float = 0.09
    aux_weight: float = 0.2


class ActionEmbedder(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.net(action.float())


class ARPredictor(nn.Module):
    def __init__(
        self,
        *,
        history_size: int,
        embed_dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, history_size, embed_dim) * 0.02)
        self.action_gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        t = emb.shape[1]
        x = emb + self.pos_embedding[:, :t] + self.action_gate(act_emb)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        return self.norm(self.transformer(x, mask=mask))


class DrivingLeWM(nn.Module):
    def __init__(self, cfg: DrivingLeWMConfig) -> None:
        super().__init__()
        if cfg.encoder_scale not in VIT_CONFIGS:
            raise ValueError(f"Unknown ViT scale {cfg.encoder_scale}; expected {sorted(VIT_CONFIGS)}")
        self.cfg = cfg
        self.encoder = self._make_vit(cfg)
        hidden_dim = int(self.encoder.config.hidden_size)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, cfg.predictor_mlp_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.predictor_mlp_dim),
            nn.Linear(cfg.predictor_mlp_dim, cfg.embed_dim),
        )
        self.action_encoder = ActionEmbedder(cfg.action_dim, cfg.embed_dim)
        self.predictor = ARPredictor(
            history_size=cfg.history_size,
            embed_dim=cfg.embed_dim,
            depth=cfg.predictor_depth,
            heads=cfg.predictor_heads,
            mlp_dim=cfg.predictor_mlp_dim,
            dropout=cfg.dropout,
        )
        self.aux_head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.GELU(),
            nn.Linear(cfg.embed_dim, 8),
        )

    @staticmethod
    def _make_vit(cfg: DrivingLeWMConfig) -> nn.Module:
        try:
            from transformers import ViTConfig, ViTModel
        except Exception as exc:
            raise RuntimeError("transformers is required for DrivingLeWM") from exc

        params = dict(VIT_CONFIGS[cfg.encoder_scale])
        params["intermediate_size"] = params["hidden_size"] * 4
        params["image_size"] = cfg.image_size
        params["patch_size"] = cfg.patch_size
        model_cfg = ViTConfig(**params)
        return ViTModel(model_cfg, add_pooling_layer=False, use_mask_token=False)

    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        b, t = pixels.shape[:2]
        flat = pixels.reshape(b * t, *pixels.shape[2:]).float()
        out = self.encoder(flat, interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0]
        emb = self.projector(cls)
        return emb.reshape(b, t, -1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        emb = self.encode_pixels(batch["pixels"])
        act_emb = self.action_encoder(batch["action"])
        ctx = emb[:, : self.cfg.history_size]
        ctx_act = act_emb[:, : self.cfg.history_size]
        pred = self.predictor(ctx, ctx_act)
        target = emb[:, 1 : self.cfg.history_size + 1].detach()
        pred = pred[:, : target.shape[1]]
        aux = self.aux_head(emb)
        return {"emb": emb, "pred_emb": pred, "target_emb": target, "aux": aux}

    @staticmethod
    def sigreg_loss(emb: torch.Tensor) -> torch.Tensor:
        flat = emb.reshape(-1, emb.shape[-1])
        mean_loss = flat.mean(dim=0).square().mean()
        std_loss = (flat.std(dim=0, unbiased=False) - 1.0).square().mean()
        return mean_loss + std_loss

    def loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self.forward(batch)
        pred_loss = F.mse_loss(out["pred_emb"], out["target_emb"])
        sigreg = self.sigreg_loss(out["emb"])

        aux_target = torch.cat(
            [
                batch["speed_mps"],
                batch["route_progress_m"],
                batch["lane_offset_m"],
                batch["heading_error_rad"],
                batch["collision"],
                batch["offroad"],
                batch["red_light"],
                batch["blocked"],
            ],
            dim=-1,
        ).float()
        aux_loss = F.smooth_l1_loss(out["aux"], aux_target)
        total = pred_loss + self.cfg.sigreg_weight * sigreg + self.cfg.aux_weight * aux_loss
        return {
            "loss": total,
            "pred_loss": pred_loss.detach(),
            "sigreg_loss": sigreg.detach(),
            "aux_loss": aux_loss.detach(),
        }

    @torch.no_grad()
    def rollout_aux(self, pixels: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        self.eval()
        emb = self.encode_pixels(pixels)
        act_emb = self.action_encoder(actions)
        ctx = emb[:, -self.cfg.history_size :]
        ctx_act = act_emb[:, -self.cfg.history_size :]
        pred = self.predictor(ctx, ctx_act)[:, -1:]
        return self.aux_head(pred).squeeze(1)
