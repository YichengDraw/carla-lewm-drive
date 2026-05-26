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

AUX_COMPONENTS = (
    "speed",
    "progress",
    "lane_offset",
    "heading_error",
    "collision",
    "offroad",
    "red_light",
    "blocked",
)


@dataclass(frozen=True)
class DrivingLeWMConfig:
    encoder_scale: str = "tiny"
    encoder_pretrained_name: str | None = None
    freeze_encoder: bool = False
    encoder_pooling: str = "cls"
    patch_size: int = 14
    image_size: int = 224
    action_dim: int = 15
    embed_dim: int = 192
    history_size: int = 3
    predictor_depth: int = 6
    predictor_heads: int = 6
    predictor_mlp_dim: int = 768
    dropout: float = 0.1
    pred_weight: float = 1.0
    sigreg_weight: float = 0.09
    use_aux_head: bool = True
    aux_weight: float = 0.2
    pred_aux_weight: float = 1.0
    aux_component_weights: tuple[float, float, float, float, float, float, float, float] | None = None
    aux_control_weight: float = 0.0
    pred_aux_control_weight: float = 1.0
    aux_control_sign_weight: float = 0.0
    aux_control_active_threshold: float = 0.02
    aux_control_sign_balance: bool = False
    aux_control_lane_gain: float = 0.35
    aux_control_heading_gain: float = 1.2
    aux_control_steer_limit: float = 0.35
    aux_lane_edge_threshold: float = 0.0
    aux_lane_edge_weight: float = 1.0
    aux_lane_underamp_weight: float = 0.0
    pred_aux_lane_underamp_weight: float = 0.0
    aux_lane_underamp_threshold: float = 0.0
    aux_lane_underamp_margin: float = 1.0
    aux_lane_underamp_balance_signs: bool = False
    action_weight: float = 0.0
    pred_action_weight: float = 1.0
    action_component_weights: tuple[float, float, float] | None = None
    action_active_steer_weight: float = 1.0
    action_active_steer_threshold: float = 0.0
    action_conflict_weight: float = 0.0
    pred_action_conflict_weight: float = 1.0
    action_teacher_only: bool = False
    progress_mode: str = "absolute"
    route_vocab_size: int = 0
    route_embed_scale: float = 1.0
    use_temporal_action_head: bool = False
    temporal_action_include_history_actions: bool = True
    temporal_action_history_noise_std: tuple[float, float, float] | None = None
    temporal_action_history_noise_prob: float = 0.0
    use_temporal_aux_head: bool = False
    temporal_aux_hidden_dim: int = 0
    temporal_aux_dropout: float = 0.1
    temporal_aux_zero_init: bool = True
    aux_lane_sign_weight: float = 0.0
    pred_aux_lane_sign_weight: float = 0.0
    aux_lane_sign_threshold: float = 0.0
    aux_lane_sign_balance: bool = False
    aux_lane_sign_loss_type: str = "margin"
    aux_lane_sign_logit_scale: float = 10.0
    aux_heading_sign_weight: float = 0.0
    pred_aux_heading_sign_weight: float = 0.0
    aux_heading_sign_threshold: float = 0.0
    aux_heading_sign_balance: bool = False
    aux_heading_sign_loss_type: str = "margin"
    aux_heading_sign_logit_scale: float = 10.0
    aux_temporal_delta_weight: float = 0.0
    aux_temporal_delta_lane_weight: float = 1.0
    aux_temporal_delta_heading_weight: float = 1.0
    aux_temporal_delta_active_lane_threshold: float = 0.0
    aux_lane_patch_mode: str = "off"
    aux_lane_patch_pooling: str = "attention"
    aux_lane_patch_hidden_dim: int = 0
    aux_lane_patch_dropout: float = 0.1
    aux_lane_patch_zero_init: bool = True


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


class SpatialPatchLaneHead(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        pooling: str,
        dropout: float,
        zero_init: bool,
    ) -> None:
        super().__init__()
        self.pooling = str(pooling).lower()
        if self.pooling not in {"attention", "mean"}:
            raise ValueError("aux_lane_patch_pooling must be one of: attention, mean")
        if self.pooling == "attention":
            self.attn = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.attn = None
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        if zero_init:
            final = self.head[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.ndim < 3:
            raise ValueError(f"Expected patch tokens with shape [..., patches, dim], got {tuple(patch_tokens.shape)}")
        if self.attn is None:
            pooled = patch_tokens.mean(dim=-2)
        else:
            weights = torch.softmax(self.attn(patch_tokens), dim=-2)
            pooled = (weights * patch_tokens).sum(dim=-2)
        return self.head(pooled)


class DrivingLeWM(nn.Module):
    def __init__(self, cfg: DrivingLeWMConfig) -> None:
        super().__init__()
        if cfg.encoder_scale not in VIT_CONFIGS:
            raise ValueError(f"Unknown ViT scale {cfg.encoder_scale}; expected {sorted(VIT_CONFIGS)}")
        self.cfg = cfg
        self.encoder = self._make_vit(cfg)
        if cfg.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad_(False)
        hidden_dim = int(self.encoder.config.hidden_size)
        projector_input_dim = self._encoder_output_dim(hidden_dim, cfg.encoder_pooling)
        self.projector = nn.Sequential(
            nn.Linear(projector_input_dim, cfg.predictor_mlp_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.predictor_mlp_dim),
            nn.Linear(cfg.predictor_mlp_dim, cfg.embed_dim),
        )
        self.action_encoder = ActionEmbedder(cfg.action_dim, cfg.embed_dim)
        self.route_embed = nn.Embedding(cfg.route_vocab_size, cfg.embed_dim) if cfg.route_vocab_size > 0 else None
        self.predictor = ARPredictor(
            history_size=cfg.history_size,
            embed_dim=cfg.embed_dim,
            depth=cfg.predictor_depth,
            heads=cfg.predictor_heads,
            mlp_dim=cfg.predictor_mlp_dim,
            dropout=cfg.dropout,
        )
        self.aux_head = (
            nn.Sequential(
                nn.LayerNorm(cfg.embed_dim),
                nn.Linear(cfg.embed_dim, cfg.embed_dim),
                nn.GELU(),
                nn.Linear(cfg.embed_dim, 8),
            )
            if cfg.use_aux_head
            else None
        )
        temporal_aux_hidden_dim = int(cfg.temporal_aux_hidden_dim) or cfg.embed_dim
        self.temporal_aux_head = (
            nn.Sequential(
                nn.LayerNorm(cfg.embed_dim * cfg.history_size),
                nn.Linear(cfg.embed_dim * cfg.history_size, temporal_aux_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(cfg.temporal_aux_dropout)),
                nn.Linear(temporal_aux_hidden_dim, 2),
            )
            if cfg.use_temporal_aux_head
            else None
        )
        if self.temporal_aux_head is not None and self.aux_head is None:
            raise ValueError("use_temporal_aux_head requires use_aux_head=true")
        if self.temporal_aux_head is not None and bool(cfg.temporal_aux_zero_init):
            final = self.temporal_aux_head[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        patch_mode = str(cfg.aux_lane_patch_mode).lower()
        if patch_mode not in {"off", "residual", "replace"}:
            raise ValueError("aux_lane_patch_mode must be one of: off, residual, replace")
        if patch_mode != "off" and self.aux_head is None:
            raise ValueError("aux_lane_patch_mode requires use_aux_head=true")
        patch_hidden_dim = int(cfg.aux_lane_patch_hidden_dim) or hidden_dim
        self.aux_lane_patch_head = (
            SpatialPatchLaneHead(
                input_dim=hidden_dim,
                hidden_dim=patch_hidden_dim,
                pooling=cfg.aux_lane_patch_pooling,
                dropout=float(cfg.aux_lane_patch_dropout),
                zero_init=bool(cfg.aux_lane_patch_zero_init),
            )
            if patch_mode != "off"
            else None
        )
        action_head_input_dim = cfg.embed_dim
        if cfg.use_temporal_action_head:
            action_head_input_dim = cfg.embed_dim * cfg.history_size
            if cfg.temporal_action_include_history_actions:
                action_head_input_dim += cfg.action_dim * cfg.history_size
        self.action_head = nn.Sequential(
            nn.LayerNorm(action_head_input_dim),
            nn.Linear(action_head_input_dim, cfg.embed_dim),
            nn.GELU(),
            nn.Linear(cfg.embed_dim, cfg.action_dim),
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
        if cfg.encoder_pretrained_name:
            return ViTModel.from_pretrained(
                cfg.encoder_pretrained_name,
                config=model_cfg,
                add_pooling_layer=False,
                ignore_mismatched_sizes=True,
            )
        return ViTModel(model_cfg, add_pooling_layer=False, use_mask_token=False)

    @staticmethod
    def _encoder_output_dim(hidden_dim: int, pooling: str) -> int:
        mode = str(pooling).lower()
        if mode in {"cls", "mean_patch"}:
            return int(hidden_dim)
        if mode == "cls_mean":
            return int(hidden_dim) * 2
        raise ValueError("encoder_pooling must be one of: cls, mean_patch, cls_mean")

    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        emb, _ = self.encode_pixels_with_patches(pixels)
        return emb

    def encode_pixels_with_patches(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = pixels.shape[:2]
        flat = pixels.reshape(b * t, *pixels.shape[2:]).float()
        out = self.encoder(flat, interpolate_pos_encoding=True)
        tokens = out.last_hidden_state
        patch_tokens = tokens[:, 1:]
        mode = str(self.cfg.encoder_pooling).lower()
        if mode == "cls":
            features = tokens[:, 0]
        elif mode == "mean_patch":
            features = patch_tokens.mean(dim=1)
        elif mode == "cls_mean":
            features = torch.cat([tokens[:, 0], patch_tokens.mean(dim=1)], dim=-1)
        else:
            raise ValueError("encoder_pooling must be one of: cls, mean_patch, cls_mean")
        emb = self.projector(features)
        return emb.reshape(b, t, -1), patch_tokens.reshape(b, t, patch_tokens.shape[-2], patch_tokens.shape[-1])

    def fuse_patch_lane_aux(self, aux: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        if self.aux_lane_patch_head is None:
            return aux
        patch_lane = self.aux_lane_patch_head(patch_tokens)
        out = aux.clone()
        mode = str(self.cfg.aux_lane_patch_mode).lower()
        if mode == "residual":
            out[..., 2:4] = out[..., 2:4] + patch_lane
        elif mode == "replace":
            out[..., 2:4] = patch_lane
        else:
            raise ValueError("aux_lane_patch_mode must be one of: off, residual, replace")
        return out

    def apply_route_condition(self, emb: torch.Tensor, route_id: torch.Tensor | int | None) -> torch.Tensor:
        if self.route_embed is None:
            return emb
        if route_id is None:
            raise ValueError("route_id is required when route_vocab_size > 0")
        b, t = emb.shape[:2]
        ids = route_id if torch.is_tensor(route_id) else torch.as_tensor(route_id, device=emb.device)
        ids = ids.to(device=emb.device)
        if ids.ndim == 0:
            ids = ids.reshape(1, 1).expand(b, t)
        elif ids.ndim == 1:
            if ids.shape[0] == b:
                ids = ids[:, None].expand(b, t)
            elif ids.shape[0] == t:
                ids = ids[None, :].expand(b, t)
            else:
                raise ValueError(f"route_id length {ids.shape[0]} does not match batch={b} or time={t}")
        elif ids.ndim == 2:
            if ids.shape != (b, t):
                raise ValueError(f"route_id shape {tuple(ids.shape)} does not match {(b, t)}")
        elif ids.ndim == 3 and ids.shape[-1] == 1:
            ids = ids.squeeze(-1)
            if ids.shape != (b, t):
                raise ValueError(f"route_id shape {tuple(ids.shape)} does not match {(b, t)}")
        else:
            raise ValueError(f"route_id must be scalar, [B], [T], [B,T], or [B,T,1], got {tuple(ids.shape)}")
        ids = ids.long().clamp(0, int(self.cfg.route_vocab_size) - 1)
        return emb + float(self.cfg.route_embed_scale) * self.route_embed(ids)

    def temporal_aux_residual(self, emb: torch.Tensor) -> torch.Tensor:
        if self.temporal_aux_head is None:
            return emb.new_zeros((*emb.shape[:2], 2))
        b, t, d = emb.shape
        history_size = int(self.cfg.history_size)
        if t <= 0:
            raise ValueError("temporal aux head requires at least one frame")
        pad = emb[:, :1].expand(b, history_size - 1, d)
        padded = torch.cat([pad, emb], dim=1)
        windows = [padded[:, i : i + history_size].reshape(b, history_size * d) for i in range(t)]
        features = torch.stack(windows, dim=1)
        return self.temporal_aux_head(features)

    def fuse_temporal_aux(self, aux: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.temporal_aux_head is None:
            return aux
        out = aux.clone()
        out[..., 2:4] = out[..., 2:4] + self.temporal_aux_residual(emb)
        return out

    def temporal_action_features(
        self,
        emb: torch.Tensor,
        action_history: torch.Tensor | None,
    ) -> torch.Tensor:
        history_size = int(self.cfg.history_size)
        if emb.shape[1] < history_size:
            raise ValueError(f"Need at least {history_size} frames for temporal action head, got {emb.shape[1]}")
        features = [emb[:, -history_size:].reshape(emb.shape[0], history_size * emb.shape[-1])]
        if self.cfg.temporal_action_include_history_actions:
            if action_history is None:
                raise ValueError("action_history is required when temporal_action_include_history_actions=true")
            if action_history.shape[1] < history_size:
                raise ValueError(
                    f"Need at least {history_size} actions for temporal action head, got {action_history.shape[1]}"
                )
            history = action_history[:, -history_size:].float()
            if self.training and self.cfg.temporal_action_history_noise_std is not None:
                if len(self.cfg.temporal_action_history_noise_std) != 3:
                    raise ValueError("temporal_action_history_noise_std must contain throttle, steer, brake std")
                prob = float(self.cfg.temporal_action_history_noise_prob)
                if prob > 0.0:
                    block = history.reshape(*history.shape[:-1], -1, 3)
                    std = history.new_tensor(self.cfg.temporal_action_history_noise_std).reshape(
                        *([1] * (block.ndim - 1)), 3
                    )
                    mask = (torch.rand(*block.shape[:-1], 1, device=block.device) < prob).to(block.dtype)
                    block = block + torch.randn_like(block) * std * mask
                    low = history.new_tensor([0.0, -1.0, 0.0]).reshape(*([1] * (block.ndim - 1)), 3)
                    high = history.new_tensor([1.0, 1.0, 1.0]).reshape(*([1] * (block.ndim - 1)), 3)
                    history = torch.clamp(block, low, high).reshape_as(history)
            features.append(history.reshape(history.shape[0], -1))
        return torch.cat(features, dim=-1)

    def action_predictions(
        self,
        emb: torch.Tensor,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.cfg.use_temporal_action_head:
            return self.action_head(self.temporal_action_features(emb, action_history))
        return self.action_head(emb)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        emb, patch_tokens = self.encode_pixels_with_patches(batch["pixels"])
        emb = self.apply_route_condition(emb, batch.get("route_id"))
        act_emb = self.action_encoder(batch["action"])
        ctx = emb[:, : self.cfg.history_size]
        ctx_act = act_emb[:, : self.cfg.history_size]
        pred = self.predictor(ctx, ctx_act)
        target = emb[:, 1 : self.cfg.history_size + 1].detach()
        pred = pred[:, : target.shape[1]]
        if self.cfg.use_temporal_action_head:
            action_history = batch["action"][:, : self.cfg.history_size]
            action = self.action_predictions(emb, action_history)
            pred_action = action.new_zeros(action.shape)
        else:
            action = self.action_predictions(emb)
            pred_action = self.action_head(pred)
        out = {
            "emb": emb,
            "pred_emb": pred,
            "target_emb": target,
            "action": action,
            "pred_action": pred_action,
        }
        if self.aux_head is not None:
            aux = self.fuse_temporal_aux(self.aux_head(emb), emb)
            out["aux"] = self.fuse_patch_lane_aux(aux, patch_tokens)
            out["pred_aux"] = self.aux_head(pred)
        return out

    @staticmethod
    def sigreg_loss(emb: torch.Tensor) -> torch.Tensor:
        flat = emb.reshape(-1, emb.shape[-1])
        mean_loss = flat.mean(dim=0).square().mean()
        std_loss = (flat.std(dim=0, unbiased=False) - 1.0).square().mean()
        return mean_loss + std_loss

    @staticmethod
    def action_conflict_loss(action: torch.Tensor) -> torch.Tensor:
        block = action.float().reshape(*action.shape[:-1], -1, 3)
        throttle = F.relu(block[..., 0])
        brake = F.relu(block[..., 2])
        return (throttle * brake).mean()

    @staticmethod
    def action_regression_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        component_weights: tuple[float, float, float] | list[float] | None = None,
        active_steer_weight: float = 1.0,
        active_steer_threshold: float = 0.0,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw = F.smooth_l1_loss(pred.float(), target.float(), reduction="none")
        block = raw.reshape(*raw.shape[:-1], -1, 3)
        mask_block = None
        if mask is not None:
            mask_block = mask.float().to(block.device)
            while mask_block.ndim < block.ndim:
                mask_block = mask_block.unsqueeze(-1)
            mask_block = mask_block.expand_as(block)
        reduce_dims = tuple(range(block.ndim - 1))
        if mask_block is None:
            component_losses = block.mean(dim=reduce_dims)
        else:
            component_losses = (block * mask_block).sum(dim=reduce_dims) / mask_block.sum(dim=reduce_dims).clamp_min(1e-8)

        if component_weights is None:
            weights = pred.new_ones(3)
        else:
            if len(component_weights) != 3:
                raise ValueError("action_component_weights must contain throttle, steer, brake weights")
            weights = pred.new_tensor(component_weights, dtype=block.dtype)
        weight_block = weights.reshape(*([1] * (block.ndim - 1)), 3).expand_as(block).clone()
        if active_steer_weight != 1.0:
            target_block = target.float().reshape(*target.shape[:-1], -1, 3)
            active = target_block[..., 1].abs() >= float(active_steer_threshold)
            weight_block[..., 1] = torch.where(
                active,
                weight_block[..., 1] * float(active_steer_weight),
                weight_block[..., 1],
            )
        if mask_block is not None:
            weight_block = weight_block * mask_block
        total = (block * weight_block).sum() / weight_block.sum().clamp_min(1e-8)
        return total, {
            "throttle": component_losses[0],
            "steer": component_losses[1],
            "brake": component_losses[2],
        }

    @staticmethod
    def progress_signal(route_progress_m: torch.Tensor, mode: str) -> torch.Tensor:
        mode = str(mode).lower()
        if mode == "absolute":
            return route_progress_m.float()
        if mode == "delta":
            progress = route_progress_m.float()
            delta = torch.zeros_like(progress)
            delta[:, 1:] = progress[:, 1:] - progress[:, :-1]
            return delta
        raise ValueError(f"Unknown progress_mode {mode!r}; expected 'absolute' or 'delta'")

    @classmethod
    def aux_target(cls, batch: dict[str, torch.Tensor], progress_mode: str) -> torch.Tensor:
        progress = cls.progress_signal(batch["route_progress_m"], progress_mode)
        return torch.cat(
            [
                batch["speed_mps"],
                progress,
                batch["lane_offset_m"],
                batch["heading_error_rad"],
                batch["collision"],
                batch["offroad"],
                batch["red_light"],
                batch["blocked"],
            ],
            dim=-1,
        ).float()

    @staticmethod
    def aux_regression_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        component_weights: tuple[float, float, float, float, float, float, float, float] | list[float] | None,
        *,
        lane_edge_threshold: float = 0.0,
        lane_edge_weight: float = 1.0,
    ) -> torch.Tensor:
        raw = F.smooth_l1_loss(pred.float(), target.float(), reduction="none")
        if component_weights is None:
            return raw.mean()
        if len(component_weights) != pred.shape[-1]:
            raise ValueError(f"aux_component_weights must contain {pred.shape[-1]} values")
        weights = pred.new_tensor(component_weights, dtype=raw.dtype)
        shaped = weights.reshape(*([1] * (raw.ndim - 1)), raw.shape[-1]).expand_as(raw).clone()
        edge_weight = float(lane_edge_weight)
        edge_threshold = max(0.0, float(lane_edge_threshold))
        if edge_weight != 1.0 and edge_threshold > 0.0:
            edge = target.float()[..., 2].abs() >= edge_threshold
            shaped[..., 2] = torch.where(edge, shaped[..., 2] * edge_weight, shaped[..., 2])
        return (raw * shaped).sum() / shaped.sum().clamp_min(1e-8)

    @staticmethod
    def aux_component_losses(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = F.smooth_l1_loss(pred.float(), target.float(), reduction="none")
        reduce_dims = tuple(range(raw.ndim - 1))
        values = raw.mean(dim=reduce_dims)
        return {name: values[i] for i, name in enumerate(AUX_COMPONENTS)}

    @staticmethod
    def aux_control_steer(
        aux: torch.Tensor,
        *,
        lane_gain: float,
        heading_gain: float,
        steer_limit: float,
    ) -> torch.Tensor:
        steer = -float(lane_gain) * aux.float()[..., 2] - float(heading_gain) * aux.float()[..., 3]
        limit = abs(float(steer_limit))
        if limit > 0.0:
            steer = torch.clamp(steer, -limit, limit)
        return steer

    @classmethod
    def aux_control_loss(
        cls,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        lane_gain: float,
        heading_gain: float,
        steer_limit: float,
        sign_weight: float,
        active_threshold: float,
        balance_signs: bool = False,
    ) -> torch.Tensor:
        pred_steer = cls.aux_control_steer(
            pred,
            lane_gain=lane_gain,
            heading_gain=heading_gain,
            steer_limit=steer_limit,
        )
        target_steer = cls.aux_control_steer(
            target,
            lane_gain=lane_gain,
            heading_gain=heading_gain,
            steer_limit=steer_limit,
        )
        loss = F.smooth_l1_loss(pred_steer, target_steer)
        if float(sign_weight) > 0.0:
            threshold = max(0.0, float(active_threshold))
            active = target_steer.abs() >= threshold
            if bool(active.any()):
                direction = torch.sign(target_steer[active])
                signed_pred = pred_steer[active] * direction
                raw_penalty = F.relu(threshold - signed_pred)
                if balance_signs:
                    group_penalties = []
                    for sign in (-1.0, 1.0):
                        group = direction == sign
                        if bool(group.any()):
                            group_penalties.append(raw_penalty[group].mean())
                    sign_penalty = torch.stack(group_penalties).mean()
                else:
                    sign_penalty = raw_penalty.mean()
                loss = loss + float(sign_weight) * sign_penalty
        return loss

    @staticmethod
    def aux_lane_underamp_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        active_threshold: float,
        margin: float,
        balance_signs: bool = False,
    ) -> torch.Tensor:
        pred_lane = pred.float()[..., 2]
        target_lane = target.float()[..., 2]
        active = target_lane.abs() >= max(0.0, float(active_threshold))
        if not bool(active.any()):
            return pred_lane.new_zeros(())
        direction = torch.sign(target_lane[active])
        signed_pred = pred_lane[active] * direction
        required = target_lane[active].abs() * max(0.0, float(margin))
        raw_penalty = F.relu(required - signed_pred)
        if balance_signs:
            group_penalties = []
            for sign in (-1.0, 1.0):
                group = direction == sign
                if bool(group.any()):
                    group_penalties.append(raw_penalty[group].mean())
            if group_penalties:
                return torch.stack(group_penalties).mean()
        return raw_penalty.mean()

    @staticmethod
    def scalar_sign_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        active_threshold: float,
        balance_signs: bool = False,
        mode: str = "margin",
        logit_scale: float = 10.0,
    ) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        threshold = max(0.0, float(active_threshold))
        active = target.abs() >= threshold
        if not bool(active.any()):
            return pred.new_zeros(())
        mode = str(mode).lower()
        direction = torch.sign(target[active])
        if mode == "margin":
            signed_pred = pred[active] * direction
            raw_penalty = F.relu(threshold - signed_pred)
        elif mode == "bce":
            labels = (direction > 0.0).to(dtype=pred.dtype)
            logits = pred[active] * max(1e-6, float(logit_scale))
            raw_penalty = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        else:
            raise ValueError("scalar sign loss mode must be one of: margin, bce")
        if balance_signs:
            group_penalties = []
            for sign in (-1.0, 1.0):
                group = direction == sign
                if bool(group.any()):
                    group_penalties.append(raw_penalty[group].mean())
            if group_penalties:
                return torch.stack(group_penalties).mean()
        return raw_penalty.mean()

    @staticmethod
    def aux_temporal_delta_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        lane_weight: float = 1.0,
        heading_weight: float = 1.0,
        active_lane_threshold: float = 0.0,
    ) -> torch.Tensor:
        if pred.shape[1] < 2:
            return pred.new_zeros(())
        pred_delta = pred.float()[:, 1:, 2:4] - pred.float()[:, :-1, 2:4]
        target_delta = target.float()[:, 1:, 2:4] - target.float()[:, :-1, 2:4]
        raw = F.smooth_l1_loss(pred_delta, target_delta, reduction="none")
        weights = pred.new_tensor([lane_weight, heading_weight], dtype=raw.dtype).reshape(1, 1, 2)
        shaped = weights.expand_as(raw).clone()
        threshold = max(0.0, float(active_lane_threshold))
        if threshold > 0.0:
            lane_pair_abs = torch.maximum(target.float()[:, 1:, 2].abs(), target.float()[:, :-1, 2].abs())
            shaped = shaped * (lane_pair_abs >= threshold).to(dtype=raw.dtype).unsqueeze(-1)
        return (raw * shaped).sum() / shaped.sum().clamp_min(1e-8)

    def loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self.forward(batch)
        pred_loss = F.mse_loss(out["pred_emb"], out["target_emb"])
        sigreg = self.sigreg_loss(out["emb"])

        zero = pred_loss.new_zeros(())
        aux_loss = zero
        pred_aux_loss = zero
        aux_total = zero
        aux_parts = {name: zero for name in AUX_COMPONENTS}
        pred_aux_parts = {name: zero for name in AUX_COMPONENTS}
        aux_control_loss = zero
        pred_aux_control_loss = zero
        aux_control_total = zero
        aux_lane_underamp_loss = zero
        pred_aux_lane_underamp_loss = zero
        aux_lane_underamp_total = zero
        aux_lane_sign_loss = zero
        pred_aux_lane_sign_loss = zero
        aux_lane_sign_total = zero
        aux_heading_sign_loss = zero
        pred_aux_heading_sign_loss = zero
        aux_heading_sign_total = zero
        aux_temporal_delta_loss = zero
        if self.aux_head is not None:
            aux_target = self.aux_target(batch, self.cfg.progress_mode)
            pred_aux_target = aux_target[:, 1 : self.cfg.history_size + 1].detach()
            aux_loss = self.aux_regression_loss(
                out["aux"],
                aux_target,
                self.cfg.aux_component_weights,
                lane_edge_threshold=self.cfg.aux_lane_edge_threshold,
                lane_edge_weight=self.cfg.aux_lane_edge_weight,
            )
            pred_aux_loss = self.aux_regression_loss(
                out["pred_aux"],
                pred_aux_target,
                self.cfg.aux_component_weights,
                lane_edge_threshold=self.cfg.aux_lane_edge_threshold,
                lane_edge_weight=self.cfg.aux_lane_edge_weight,
            )
            aux_total = aux_loss + self.cfg.pred_aux_weight * pred_aux_loss
            aux_parts = self.aux_component_losses(out["aux"], aux_target)
            pred_aux_parts = self.aux_component_losses(out["pred_aux"], pred_aux_target)
            if float(self.cfg.aux_control_weight) > 0.0:
                aux_control_loss = self.aux_control_loss(
                    out["aux"],
                    aux_target,
                    lane_gain=self.cfg.aux_control_lane_gain,
                    heading_gain=self.cfg.aux_control_heading_gain,
                    steer_limit=self.cfg.aux_control_steer_limit,
                    sign_weight=self.cfg.aux_control_sign_weight,
                    active_threshold=self.cfg.aux_control_active_threshold,
                    balance_signs=self.cfg.aux_control_sign_balance,
                )
                pred_aux_control_loss = self.aux_control_loss(
                    out["pred_aux"],
                    pred_aux_target,
                    lane_gain=self.cfg.aux_control_lane_gain,
                    heading_gain=self.cfg.aux_control_heading_gain,
                    steer_limit=self.cfg.aux_control_steer_limit,
                    sign_weight=self.cfg.aux_control_sign_weight,
                    active_threshold=self.cfg.aux_control_active_threshold,
                    balance_signs=self.cfg.aux_control_sign_balance,
                )
                aux_control_total = aux_control_loss + self.cfg.pred_aux_control_weight * pred_aux_control_loss
            if float(self.cfg.aux_lane_underamp_weight) > 0.0:
                aux_lane_underamp_loss = self.aux_lane_underamp_loss(
                    out["aux"],
                    aux_target,
                    active_threshold=self.cfg.aux_lane_underamp_threshold,
                    margin=self.cfg.aux_lane_underamp_margin,
                    balance_signs=self.cfg.aux_lane_underamp_balance_signs,
                )
                pred_aux_lane_underamp_loss = self.aux_lane_underamp_loss(
                    out["pred_aux"],
                    pred_aux_target,
                    active_threshold=self.cfg.aux_lane_underamp_threshold,
                    margin=self.cfg.aux_lane_underamp_margin,
                    balance_signs=self.cfg.aux_lane_underamp_balance_signs,
                )
                aux_lane_underamp_total = (
                    aux_lane_underamp_loss
                    + self.cfg.pred_aux_lane_underamp_weight * pred_aux_lane_underamp_loss
                )
            if float(self.cfg.aux_lane_sign_weight) > 0.0:
                aux_lane_sign_loss = self.scalar_sign_loss(
                    out["aux"][..., 2],
                    aux_target[..., 2],
                    active_threshold=self.cfg.aux_lane_sign_threshold,
                    balance_signs=self.cfg.aux_lane_sign_balance,
                    mode=self.cfg.aux_lane_sign_loss_type,
                    logit_scale=self.cfg.aux_lane_sign_logit_scale,
                )
                pred_aux_lane_sign_loss = self.scalar_sign_loss(
                    out["pred_aux"][..., 2],
                    pred_aux_target[..., 2],
                    active_threshold=self.cfg.aux_lane_sign_threshold,
                    balance_signs=self.cfg.aux_lane_sign_balance,
                    mode=self.cfg.aux_lane_sign_loss_type,
                    logit_scale=self.cfg.aux_lane_sign_logit_scale,
                )
                aux_lane_sign_total = (
                    aux_lane_sign_loss
                    + self.cfg.pred_aux_lane_sign_weight * pred_aux_lane_sign_loss
                )
            if float(self.cfg.aux_heading_sign_weight) > 0.0:
                aux_heading_sign_loss = self.scalar_sign_loss(
                    out["aux"][..., 3],
                    aux_target[..., 3],
                    active_threshold=self.cfg.aux_heading_sign_threshold,
                    balance_signs=self.cfg.aux_heading_sign_balance,
                    mode=self.cfg.aux_heading_sign_loss_type,
                    logit_scale=self.cfg.aux_heading_sign_logit_scale,
                )
                pred_aux_heading_sign_loss = self.scalar_sign_loss(
                    out["pred_aux"][..., 3],
                    pred_aux_target[..., 3],
                    active_threshold=self.cfg.aux_heading_sign_threshold,
                    balance_signs=self.cfg.aux_heading_sign_balance,
                    mode=self.cfg.aux_heading_sign_loss_type,
                    logit_scale=self.cfg.aux_heading_sign_logit_scale,
                )
                aux_heading_sign_total = (
                    aux_heading_sign_loss
                    + self.cfg.pred_aux_heading_sign_weight * pred_aux_heading_sign_loss
                )
            if float(self.cfg.aux_temporal_delta_weight) > 0.0:
                aux_temporal_delta_loss = self.aux_temporal_delta_loss(
                    out["aux"],
                    aux_target,
                    lane_weight=self.cfg.aux_temporal_delta_lane_weight,
                    heading_weight=self.cfg.aux_temporal_delta_heading_weight,
                    active_lane_threshold=self.cfg.aux_temporal_delta_active_lane_threshold,
                )

        action_target_all = batch.get("teacher_action", batch["action"]).float()
        action_mask_all = batch.get("teacher_action_mask")
        action_target = action_target_all[:, -1] if self.cfg.use_temporal_action_head else action_target_all
        pred_action_target = action_target_all[:, 1 : self.cfg.history_size + 1].detach()
        action_mask = None
        pred_action_mask = None
        if self.cfg.action_teacher_only:
            if action_mask_all is None:
                action_mask_all = action_target_all.new_zeros((*action_target_all.shape[:-1], 1))
            action_mask_all = action_mask_all.to(action_target_all.device).float()
            action_mask = action_mask_all[:, -1] if self.cfg.use_temporal_action_head else action_mask_all
            pred_action_mask = action_mask_all[:, 1 : self.cfg.history_size + 1].detach()
        action_loss, action_parts = self.action_regression_loss(
            out["action"],
            action_target,
            component_weights=self.cfg.action_component_weights,
            active_steer_weight=self.cfg.action_active_steer_weight,
            active_steer_threshold=self.cfg.action_active_steer_threshold,
            mask=action_mask,
        )
        if self.cfg.use_temporal_action_head:
            pred_action_loss = zero
            pred_action_parts = {"throttle": zero, "steer": zero, "brake": zero}
        else:
            pred_action_loss, pred_action_parts = self.action_regression_loss(
                out["pred_action"],
                pred_action_target,
                component_weights=self.cfg.action_component_weights,
                active_steer_weight=self.cfg.action_active_steer_weight,
                active_steer_threshold=self.cfg.action_active_steer_threshold,
                mask=pred_action_mask,
            )
        action_total = action_loss + self.cfg.pred_action_weight * pred_action_loss
        action_conflict = self.action_conflict_loss(out["action"])
        pred_action_conflict = self.action_conflict_loss(out["pred_action"])
        action_conflict_total = (
            action_conflict + self.cfg.pred_action_conflict_weight * pred_action_conflict
        )

        total = (
            self.cfg.pred_weight * pred_loss
            + self.cfg.sigreg_weight * sigreg
            + self.cfg.aux_weight * aux_total
            + self.cfg.aux_control_weight * aux_control_total
            + self.cfg.aux_lane_underamp_weight * aux_lane_underamp_total
            + self.cfg.aux_lane_sign_weight * aux_lane_sign_total
            + self.cfg.aux_heading_sign_weight * aux_heading_sign_total
            + self.cfg.aux_temporal_delta_weight * aux_temporal_delta_loss
            + self.cfg.action_weight * action_total
            + self.cfg.action_conflict_weight * action_conflict_total
        )
        return {
            "loss": total,
            "pred_loss": pred_loss.detach(),
            "sigreg_loss": sigreg.detach(),
            "aux_loss": aux_loss.detach(),
            "pred_aux_loss": pred_aux_loss.detach(),
            **{f"aux_{name}_loss": value.detach() for name, value in aux_parts.items()},
            **{f"pred_aux_{name}_loss": value.detach() for name, value in pred_aux_parts.items()},
            "aux_control_loss": aux_control_loss.detach(),
            "pred_aux_control_loss": pred_aux_control_loss.detach(),
            "aux_lane_underamp_loss": aux_lane_underamp_loss.detach(),
            "pred_aux_lane_underamp_loss": pred_aux_lane_underamp_loss.detach(),
            "aux_lane_sign_loss": aux_lane_sign_loss.detach(),
            "pred_aux_lane_sign_loss": pred_aux_lane_sign_loss.detach(),
            "aux_heading_sign_loss": aux_heading_sign_loss.detach(),
            "pred_aux_heading_sign_loss": pred_aux_heading_sign_loss.detach(),
            "aux_temporal_delta_loss": aux_temporal_delta_loss.detach(),
            "action_loss": action_loss.detach(),
            "pred_action_loss": pred_action_loss.detach(),
            "action_throttle_loss": action_parts["throttle"].detach(),
            "action_steer_loss": action_parts["steer"].detach(),
            "action_brake_loss": action_parts["brake"].detach(),
            "pred_action_throttle_loss": pred_action_parts["throttle"].detach(),
            "pred_action_steer_loss": pred_action_parts["steer"].detach(),
            "pred_action_brake_loss": pred_action_parts["brake"].detach(),
            "action_conflict_loss": action_conflict.detach(),
            "pred_action_conflict_loss": pred_action_conflict.detach(),
        }

    @torch.no_grad()
    def rollout_aux(
        self,
        pixels: torch.Tensor,
        actions: torch.Tensor,
        route_id: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        if self.aux_head is None:
            raise RuntimeError("rollout_aux requires use_aux_head=true; use policy_action for no-aux action models")
        self.eval()
        emb = self.apply_route_condition(self.encode_pixels(pixels), route_id)
        act_emb = self.action_encoder(actions)
        ctx = emb[:, -self.cfg.history_size :]
        ctx_act = act_emb[:, -self.cfg.history_size :]
        pred = self.predictor(ctx, ctx_act)[:, -1:]
        return self.aux_head(pred).squeeze(1)

    @torch.no_grad()
    def policy_action(
        self,
        pixels: torch.Tensor,
        route_id: torch.Tensor | int | None = None,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.eval()
        emb = self.apply_route_condition(self.encode_pixels(pixels), route_id)
        if self.cfg.use_temporal_action_head:
            return self.action_predictions(emb, action_history)
        return self.action_head(emb[:, -1])

    @torch.no_grad()
    def perceive_aux(self, pixels: torch.Tensor, route_id: torch.Tensor | int | None = None) -> torch.Tensor:
        if self.aux_head is None:
            raise RuntimeError("perceive_aux requires use_aux_head=true")
        self.eval()
        emb, patch_tokens = self.encode_pixels_with_patches(pixels)
        emb = self.apply_route_condition(emb, route_id)
        aux_all = self.fuse_temporal_aux(self.aux_head(emb), emb)
        aux = aux_all[:, -1]
        return self.fuse_patch_lane_aux(aux, patch_tokens[:, -1])
