import torch
import torch.nn as nn

from nested_learning.hope.block import (
    HOPEAttentionBlock,
    HOPEAttentionBlockConfig,
    HOPESelfModBlock,
    HOPESelfModBlockConfig,
)
from nested_learning.levels import LevelSpec


class HopeEncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        variant: str = "hope_attention",
        cms_periods=(1, 4),
        hidden_multiplier: int = 4,
        cms_online_updates: bool = False,
    ):
        super().__init__()

        cms_levels = [
            LevelSpec(name=f"cms_{p}", update_period=p)
            for p in cms_periods
        ]

        if variant == "hope_attention":
            self.block = HOPEAttentionBlock(
                HOPEAttentionBlockConfig(
                    dim=dim,
                    heads=heads,
                    cms_levels=cms_levels,
                    cms_hidden_multiplier=hidden_multiplier,
                    cms_online_updates=cms_online_updates,
                    cms_use_layernorm=True,
                    activation="gelu",
                    qk_l2_norm=False,
                    local_conv_window=None,
                )
            )

        elif variant == "hope_selfmod":
            self.block = HOPESelfModBlock(
                HOPESelfModBlockConfig(
                    dim=dim,
                    cms_levels=cms_levels,
                    cms_hidden_multiplier=hidden_multiplier,
                    cms_online_updates=cms_online_updates,
                    selfmod_online_updates=False,
                    cms_use_layernorm=True,
                    activation="gelu",
                    qk_l2_norm=True,
                    selfmod_local_conv_window=None,
                )
            )

        else:
            raise ValueError(f"unknown variant: {variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 2
    T = 16
    D = 128
    H = 8

    print("=" * 80)
    print("Testing HopeEncoderBlock")
    print(f"device = {device}")
    print(f"torch  = {torch.__version__}")
    print("=" * 80)

    x = torch.randn(B, T, D, device=device)

    model = HopeEncoderBlock(
        dim=D,
        heads=H,
        variant="hope_attention",
        cms_periods=(1, 4),
        hidden_multiplier=4,
        cms_online_updates=False,
    ).to(device)

    print(model)
    print("input :", x.shape)

    y = model(x)

    print("output:", y.shape)

    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape, f"shape mismatch: x={x.shape}, y={y.shape}"

    loss = y.mean()
    loss.backward()

    print("backward: OK")
    print("test    : OK")