#!/usr/bin/env python
"""MarineSim model backbone.

Official model mainline:
- marine3d_transformer
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


MARINE_MODEL_NAMES = ("marine3d_transformer",)


def _resolve_num_heads(embed_dim: int, requested_heads: int) -> int:
    heads = max(1, min(int(requested_heads), int(embed_dim)))
    while embed_dim % heads != 0 and heads > 1:
        heads -= 1
    return max(1, heads)


def _downsample_visible_mask(
    visible_mask: torch.Tensor,
    levels: int = 2,
    keep_channel: bool = False,
) -> torch.Tensor:
    vm = visible_mask.float()
    for _ in range(max(levels, 0)):
        vm = F.max_pool3d(vm, kernel_size=2, stride=2)
    vm = (vm > 0).float()
    if keep_channel:
        return vm
    return (vm.sum(dim=1, keepdim=True) > 0).float()


def _downsample_masked_input(
    input_tensor: torch.Tensor,
    visible_mask: torch.Tensor,
    levels: int = 2,
) -> torch.Tensor:
    x = input_tensor * visible_mask
    vm = visible_mask.float()
    for _ in range(max(levels, 0)):
        x = F.avg_pool3d(x, kernel_size=2, stride=2)
        vm = F.avg_pool3d(vm, kernel_size=2, stride=2)
    return torch.where(vm > 0, x / torch.clamp(vm, min=1e-6), torch.zeros_like(x))


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=8 if out_ch >= 8 else 1, num_channels=out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8 if out_ch >= 8 else 1, num_channels=out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = F.relu(self.norm2(self.conv2(x)), inplace=True)
        return x


class AxisAttentionBlock3D(nn.Module):
    """Axis-wise attention skeleton for future H/V factorized experiments.

    Supported axis names:
    - "depth": sequence over depth for each (h, w)
    - "height": sequence over height for each (d, w)
    - "width": sequence over width for each (d, h)
    """

    def __init__(self, channels: int, axis: str, num_heads: int = 4):
        super().__init__()
        if axis not in {"depth", "height", "width"}:
            raise ValueError(f"Unsupported axis: {axis}")
        self.axis = axis
        self.channels = channels
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape

        if self.axis == "depth":
            seq = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, d, c)
            mixed, _ = self.attn(self.norm(seq), self.norm(seq), self.norm(seq))
            mixed = self.proj(mixed).reshape(b, h, w, d, c).permute(0, 4, 3, 1, 2)
            return x + mixed

        if self.axis == "height":
            seq = x.permute(0, 2, 4, 3, 1).reshape(b * d * w, h, c)
            mixed, _ = self.attn(self.norm(seq), self.norm(seq), self.norm(seq))
            mixed = self.proj(mixed).reshape(b, d, w, h, c).permute(0, 4, 1, 3, 2)
            return x + mixed

        seq = x.permute(0, 2, 3, 4, 1).reshape(b * d * h, w, c)
        mixed, _ = self.attn(self.norm(seq), self.norm(seq), self.norm(seq))
        mixed = self.proj(mixed).reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)
        return x + mixed


class ChannelAttentionBlock3D(nn.Module):
    """Lightweight channel-mixing block (variable-attention placeholder)."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // max(reduction, 1), 8)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.pool(x)
        g = F.relu(self.fc1(g), inplace=True)
        g = torch.sigmoid(self.fc2(g))
        return x + x * g


class BottleneckMixer(nn.Module):
    """Lightweight bottleneck mixer hook.

    mode:
    - "none": identity
    - "grouped_conv": grouped 3D convolution mixing (default lightweight hook)
    - "plain_attention": lightweight token attention without visibility mask
    - "masked_attention": lightweight token attention with visibility-aware mask
    - "h_attention"/"v_attention"/"c_attention"/"hv_attention"/"hvc_attention":
      reserved runnable skeletons for future grouped H/V/C expansion
    """

    def __init__(self, channels: int, mode: str = "grouped_conv", num_heads: int = 4):
        super().__init__()
        self.mode = mode
        self.channels = channels
        self.num_heads = num_heads
        self.norm: Optional[nn.LayerNorm] = None
        self.attn: Optional[nn.MultiheadAttention] = None
        self.proj: Optional[nn.Linear] = None
        self.mix: Optional[nn.Module] = None

        if mode == "grouped_conv":
            groups = 4 if channels % 4 == 0 else 1
            self.mix = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=groups)
        elif mode in {"plain_attention", "masked_attention"}:
            self.norm = nn.LayerNorm(channels)
            self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
            self.proj = nn.Linear(channels, channels)
        elif mode in {"h_attention", "v_attention", "c_attention", "hv_attention", "hvc_attention"}:
            blocks: list[nn.Module] = []
            if mode in {"h_attention", "hv_attention", "hvc_attention"}:
                blocks.append(AxisAttentionBlock3D(channels=channels, axis="height", num_heads=num_heads))
            if mode in {"v_attention", "hv_attention", "hvc_attention"}:
                blocks.append(AxisAttentionBlock3D(channels=channels, axis="width", num_heads=num_heads))
            if mode in {"c_attention", "hvc_attention"}:
                blocks.append(ChannelAttentionBlock3D(channels=channels))
            self.mix = nn.Sequential(*blocks)
        elif mode == "none":
            self.mix = None
        else:
            raise ValueError(f"Unsupported mixer mode: {mode}")

    def _build_attention_masks(
        self,
        x: torch.Tensor,
        bottleneck_visible_mask: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.mode != "masked_attention" or bottleneck_visible_mask is None:
            return None, None

        b, _, d, h, w = x.shape
        vm = bottleneck_visible_mask.view(b, -1) > 0
        key_padding_mask = ~vm

        # Do not let missing queries attend missing keys.
        q_missing = (~vm).unsqueeze(2)
        k_missing = (~vm).unsqueeze(1)
        attn_mask = q_missing & k_missing
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attn_mask = attn_mask.reshape(b * self.num_heads, d * h * w, d * h * w)

        all_masked = key_padding_mask.all(dim=1)
        if bool(all_masked.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
            attn_mask = attn_mask.clone()
            bad_index = torch.where(all_masked)[0]
            for bi in bad_index.tolist():
                start = bi * self.num_heads
                end = (bi + 1) * self.num_heads
                attn_mask[start:end] = False

        return key_padding_mask, attn_mask

    def forward(self, x: torch.Tensor, bottleneck_visible_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.mode == "none":
            return x

        if self.mode == "grouped_conv":
            return x + F.relu(self.mix(x), inplace=True)

        if self.mode in {"h_attention", "v_attention", "c_attention", "hv_attention", "hvc_attention"}:
            return self.mix(x)

        # plain/masked attention branch
        b, c, d, h, w = x.shape
        tokens = x.view(b, c, d * h * w).transpose(1, 2)  # [B,N,C]
        tokens_n = self.norm(tokens)

        key_padding_mask, attn_mask = self._build_attention_masks(x=x, bottleneck_visible_mask=bottleneck_visible_mask)
        mixed, _ = self.attn(tokens_n, tokens_n, tokens_n, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        mixed = self.proj(mixed)
        mixed = mixed.transpose(1, 2).view(b, c, d, h, w)
        return x + mixed


class SpatialPositionalEncoding3D(nn.Module):
    """Lightweight continuous 3D positional encoding for bottleneck tokens."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 2, 16)
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, d: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        z = torch.linspace(-1.0, 1.0, steps=d, device=device, dtype=dtype)
        y = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        coords = torch.stack([zz, yy, xx], dim=-1).reshape(d * h * w, 3)
        pos = self.mlp(coords)
        return pos.unsqueeze(0)  # [1, N, C]


class TransformerBottleneckBlock3D(nn.Module):
    """Transformer block with horizontal, depth-axis and variable-relation mixing."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        ffn_mult: int = 2,
        dropout: float = 0.0,
        use_depth_axis: bool = True,
        use_channel_relation: bool = True,
        spatial_mode: str = "depth_slice",
        num_variables: int = 8,
    ):
        super().__init__()
        if spatial_mode not in {"global", "depth_slice"}:
            raise ValueError(f"Unsupported spatial_mode: {spatial_mode}")

        heads = _resolve_num_heads(embed_dim=channels, requested_heads=num_heads)
        hidden = channels * max(1, ffn_mult)
        self.spatial_mode = spatial_mode
        self.num_variables = max(1, int(num_variables))

        self.token_norm = nn.LayerNorm(channels)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            batch_first=True,
            dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

        self.use_depth_axis = bool(use_depth_axis)
        if self.use_depth_axis:
            self.depth_norm = nn.LayerNorm(channels)
            self.depth_attn = nn.MultiheadAttention(
                embed_dim=channels,
                num_heads=heads,
                batch_first=True,
                dropout=dropout,
            )
            self.depth_proj = nn.Linear(channels, channels)

        self.use_channel_relation = bool(use_channel_relation)
        if self.use_channel_relation:
            var_embed = max(16, min(channels, 64))
            var_heads = _resolve_num_heads(embed_dim=var_embed, requested_heads=num_heads)
            self.variable_token = nn.Linear(1, var_embed)
            self.variable_norm = nn.LayerNorm(var_embed)
            self.variable_attn = nn.MultiheadAttention(
                embed_dim=var_embed,
                num_heads=var_heads,
                batch_first=True,
                dropout=dropout,
            )
            self.variable_out = nn.Linear(var_embed, 1)
            self.feature_to_variable = nn.Conv3d(channels, self.num_variables, kernel_size=1, bias=False)
            self.variable_to_feature = nn.Linear(self.num_variables, channels)

    @staticmethod
    def _sanitize_key_padding_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        bad = mask.all(dim=1)
        if bool(bad.any()):
            mask = mask.clone()
            mask[bad] = False
        return mask

    def _spatial_attention(
        self,
        x: torch.Tensor,
        position_visible_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, c, d, h, w = x.shape
        if self.spatial_mode == "global":
            seq = x.view(b, c, d * h * w).transpose(1, 2)  # [B,N,C]
            seq_n = self.token_norm(seq)
            key_padding_mask = None
            if position_visible_mask is not None:
                token_visible = position_visible_mask.view(b, -1).bool()
                key_padding_mask = self._sanitize_key_padding_mask(~token_visible)
            mixed, _ = self.spatial_attn(seq_n, seq_n, seq_n, key_padding_mask=key_padding_mask)
            seq = seq + self.drop(mixed)
            seq = seq + self.ffn(self.ffn_norm(seq))
            return seq.transpose(1, 2).view(b, c, d, h, w)

        # Depth-slice mode: attend only inside each depth layer over HxW tokens.
        seq = x.permute(0, 2, 3, 4, 1).reshape(b * d, h * w, c)  # [B*D,HW,C]
        seq_n = self.token_norm(seq)
        key_padding_mask = None
        if position_visible_mask is not None:
            vm_hw = position_visible_mask.squeeze(1).reshape(b * d, h * w).bool()
            key_padding_mask = self._sanitize_key_padding_mask(~vm_hw)
        mixed, _ = self.spatial_attn(seq_n, seq_n, seq_n, key_padding_mask=key_padding_mask)
        seq = seq + self.drop(mixed)
        seq = seq + self.ffn(self.ffn_norm(seq))
        return seq.reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)

    def _depth_attention(
        self,
        x: torch.Tensor,
        position_visible_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, c, d, h, w = x.shape
        seq = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, d, c)
        seq_n = self.depth_norm(seq)

        key_padding_mask = None
        if position_visible_mask is not None:
            vm_d = position_visible_mask.squeeze(1).amax(dim=(2, 3)).bool()  # [B,D]
            vm_hw = vm_d[:, None, None, :].expand(-1, h, w, -1).reshape(b * h * w, d)
            key_padding_mask = self._sanitize_key_padding_mask(~vm_hw)

        mixed, _ = self.depth_attn(
            seq_n,
            seq_n,
            seq_n,
            key_padding_mask=key_padding_mask,
        )
        mixed = self.depth_proj(mixed)
        mixed = mixed.reshape(b, h, w, d, c).permute(0, 4, 3, 1, 2)
        return x + self.drop(mixed)

    def _channel_relation(
        self,
        x: torch.Tensor,
        variable_visible_mask: Optional[torch.Tensor],
        variable_context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, c, d, h, w = x.shape
        if variable_context is None:
            var_src = self.feature_to_variable(x)  # [B,V,D,H,W]
        else:
            var_src = variable_context
            if var_src.shape[-3:] != (d, h, w):
                var_src = F.interpolate(var_src, size=(d, h, w), mode="trilinear", align_corners=False)
            if int(var_src.shape[1]) != self.num_variables:
                raise ValueError(
                    f"variable_context channels mismatch: expected {self.num_variables}, got {int(var_src.shape[1])}"
                )

        if variable_visible_mask is None:
            vm = torch.ones((b, self.num_variables, d, h, w), device=x.device, dtype=x.dtype)
        else:
            vm = variable_visible_mask.float()
            if vm.shape[-3:] != (d, h, w):
                vm = F.interpolate(vm, size=(d, h, w), mode="nearest")
            if int(vm.shape[1]) != self.num_variables:
                raise ValueError(
                    f"variable_visible_mask channels mismatch: expected {self.num_variables}, got {int(vm.shape[1])}"
                )

        visible_count = vm.sum(dim=(2, 3, 4))  # [B,V]
        den = torch.clamp(visible_count, min=1.0)
        pooled = (var_src * vm).sum(dim=(2, 3, 4)) / den  # [B,V]

        token = self.variable_token(pooled.unsqueeze(-1))  # [B,V,E]
        token_n = self.variable_norm(token)
        var_visible = visible_count > 0
        key_padding_mask = self._sanitize_key_padding_mask(~var_visible)
        mixed, _ = self.variable_attn(token_n, token_n, token_n, key_padding_mask=key_padding_mask)
        gate_var = torch.sigmoid(self.variable_out(mixed).squeeze(-1))  # [B,V]
        gate_feature = torch.sigmoid(self.variable_to_feature(gate_var)).view(b, c, 1, 1, 1)
        return x + x * gate_feature

    def forward(
        self,
        x: torch.Tensor,
        pos_tokens: torch.Tensor,
        position_visible_mask: Optional[torch.Tensor],
        variable_visible_mask: Optional[torch.Tensor],
        position_visible_weight: Optional[torch.Tensor],
        variable_context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, c, d, h, w = x.shape
        tokens = x.view(b, c, d * h * w).transpose(1, 2)
        tokens = tokens + pos_tokens
        x = tokens.transpose(1, 2).view(b, c, d, h, w)

        x = self._spatial_attention(x, position_visible_mask=position_visible_mask)
        if self.use_depth_axis:
            x = self._depth_attention(x, position_visible_mask=position_visible_mask)
        if self.use_channel_relation:
            x = self._channel_relation(
                x,
                variable_visible_mask=variable_visible_mask,
                variable_context=variable_context,
            )
        if position_visible_weight is not None:
            x = x * position_visible_weight
        return x


class FullTransformerBottleneck3D(nn.Module):
    """Mask-aware Transformer bottleneck for full Step6 model variant."""

    def __init__(
        self,
        channels: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_mult: int = 2,
        dropout: float = 0.0,
        mask_aware: bool = True,
        use_depth_axis: bool = True,
        use_channel_relation: bool = True,
        spatial_mode: str = "depth_slice",
        num_variables: int = 8,
    ):
        super().__init__()
        self.mask_aware = bool(mask_aware)
        self.pos_encoding = SpatialPositionalEncoding3D(channels)
        self.num_variables = max(1, int(num_variables))
        self.layers = nn.ModuleList(
            [
                TransformerBottleneckBlock3D(
                    channels=channels,
                    num_heads=num_heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    use_depth_axis=use_depth_axis,
                    use_channel_relation=use_channel_relation,
                    spatial_mode=spatial_mode,
                    num_variables=self.num_variables,
                )
                for _ in range(max(1, num_layers))
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        bottleneck_visible_mask: Optional[torch.Tensor] = None,
        bottleneck_visible_mask_by_var: Optional[torch.Tensor] = None,
        bottleneck_variable_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, _, d, h, w = x.shape
        vm_pos = None
        vm_var = None
        vm_weight = None
        if self.mask_aware:
            if bottleneck_visible_mask_by_var is not None:
                vm_var = (bottleneck_visible_mask_by_var > 0).float()
                if int(vm_var.shape[1]) != self.num_variables:
                    raise ValueError(
                        f"bottleneck_visible_mask_by_var channels mismatch: expected {self.num_variables}, got {int(vm_var.shape[1])}"
                    )
                if vm_var.shape[-3:] != (d, h, w):
                    vm_var = F.interpolate(vm_var, size=(d, h, w), mode="nearest")

            if bottleneck_visible_mask is not None:
                vm_pos = (bottleneck_visible_mask > 0).float()
                if vm_pos.shape[-3:] != (d, h, w):
                    vm_pos = F.interpolate(vm_pos, size=(d, h, w), mode="nearest")
            elif vm_var is not None:
                vm_pos = (vm_var.sum(dim=1, keepdim=True) > 0).float()

            if vm_var is not None:
                vm_weight = vm_var.mean(dim=1, keepdim=True)
            else:
                vm_weight = vm_pos

            if vm_weight is not None:
                x = x * vm_weight

        pos = self.pos_encoding(d=d, h=h, w=w, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x = layer(
                x,
                pos_tokens=pos,
                position_visible_mask=vm_pos,
                variable_visible_mask=vm_var,
                position_visible_weight=vm_weight,
                variable_context=bottleneck_variable_context,
            )
        return x


class LegacyUNet3D(nn.Module):
    """Deprecated legacy U-Net branch.

    Kept only as quarantined legacy implementation. It is not part of the
    official MarineSim model mainline and is never selected by training entry.
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 8,
        base_channels: int = 32,
        mixer_mode: str = "grouped_conv",
        mixer_num_heads: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.enc1 = ConvBlock3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock3D(base_channels * 2, base_channels * 4)
        self.mixer = BottleneckMixer(channels=base_channels * 4, mode=mixer_mode, num_heads=mixer_num_heads)

        self.up1 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_channels * 4, base_channels * 2)

        self.up2 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_channels * 2, base_channels)

        self.head = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] == ref.shape[2:]:
            return x
        return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, input_tensor: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([input_tensor, visible_mask], dim=1)

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        # Downsample visibility mask to bottleneck scale for masked-attention compatibility.
        vm_b = _downsample_visible_mask(visible_mask, levels=2)

        b = self.mixer(b, bottleneck_visible_mask=vm_b)

        u1 = self.up1(b)
        u1 = self._resize_like(u1, e2)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))

        u2 = self.up2(d1)
        u2 = self._resize_like(u2, e1)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))

        out = self.head(d2)
        return out


class Marine3DTransformerCore(nn.Module):
    """3D U-Net encoder/decoder with a full transformer bottleneck."""

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 8,
        base_channels: int = 32,
        transformer_num_layers: int = 2,
        transformer_num_heads: int = 4,
        transformer_ffn_mult: int = 2,
        transformer_dropout: float = 0.0,
        transformer_mask_aware: bool = True,
        transformer_use_depth_axis: bool = True,
        transformer_use_channel_relation: bool = True,
        transformer_spatial_mode: str = "depth_slice",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.enc1 = ConvBlock3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock3D(base_channels * 2, base_channels * 4)
        self.transformer = FullTransformerBottleneck3D(
            channels=base_channels * 4,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            ffn_mult=transformer_ffn_mult,
            dropout=transformer_dropout,
            mask_aware=transformer_mask_aware,
            use_depth_axis=transformer_use_depth_axis,
            use_channel_relation=transformer_use_channel_relation,
            spatial_mode=transformer_spatial_mode,
            num_variables=out_channels,
        )

        self.up1 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_channels * 4, base_channels * 2)

        self.up2 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_channels * 2, base_channels)

        self.head = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] == ref.shape[2:]:
            return x
        return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, input_tensor: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([input_tensor, visible_mask], dim=1)

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)
        vm_b_var = _downsample_visible_mask(visible_mask, levels=2, keep_channel=True)
        vm_b_pos = (vm_b_var.sum(dim=1, keepdim=True) > 0).float()
        var_ctx_b = _downsample_masked_input(input_tensor=input_tensor, visible_mask=visible_mask, levels=2)
        b = self.transformer(
            b,
            bottleneck_visible_mask=vm_b_pos,
            bottleneck_visible_mask_by_var=vm_b_var,
            bottleneck_variable_context=var_ctx_b,
        )

        u1 = self.up1(b)
        u1 = self._resize_like(u1, e2)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))

        u2 = self.up2(d1)
        u2 = self._resize_like(u2, e1)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))

        out = self.head(d2)
        return out


def build_marine_model(
    model_name: str = "marine3d_transformer",
    in_channels: int = 16,
    out_channels: int = 8,
    base_channels: int = 32,
    mixer_mode: str = "grouped_conv",
    mixer_num_heads: int = 4,
    transformer_num_layers: int = 2,
    transformer_num_heads: int = 4,
    transformer_ffn_mult: int = 2,
    transformer_dropout: float = 0.0,
    transformer_mask_aware: bool = True,
    transformer_use_depth_axis: bool = True,
    transformer_use_channel_relation: bool = True,
    transformer_spatial_mode: str = "depth_slice",
) -> nn.Module:
    if model_name != "marine3d_transformer":
        raise ValueError(f"Unsupported model_name: {model_name}. Supported: {MARINE_MODEL_NAMES}")
    return Marine3DTransformerCore(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        transformer_num_layers=transformer_num_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_ffn_mult=transformer_ffn_mult,
        transformer_dropout=transformer_dropout,
        transformer_mask_aware=transformer_mask_aware,
        transformer_use_depth_axis=transformer_use_depth_axis,
        transformer_use_channel_relation=transformer_use_channel_relation,
        transformer_spatial_mode=transformer_spatial_mode,
    )


