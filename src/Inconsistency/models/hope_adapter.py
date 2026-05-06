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
                    cms_online_updates=False,  # 先關掉，避免訓練不穩
                )
            )
        elif variant == "hope_selfmod":
            self.block = HOPESelfModBlock(
                HOPESelfModBlockConfig(
                    dim=dim,
                    cms_levels=cms_levels,
                    cms_hidden_multiplier=hidden_multiplier,
                    cms_online_updates=False,
                    selfmod_online_updates=False,
                )
            )
        else:
            raise ValueError(f"unknown variant: {variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)