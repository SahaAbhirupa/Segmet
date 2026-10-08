"""Networks: SegMet (SegNet-style encoder-decoder) and CNN-regression baselines."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import IMG, N_CELLS, N_CLASSES, cell_bins


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class CellPool(nn.Module):
    """Average a full-resolution map over each unit cell's pixel footprint (224 -> 20).

    Same bins as adaptive average pooling, written as two matmuls so it also runs
    on MPS (which lacks non-divisible adaptive pooling)."""

    def __init__(self, n_cells: int = N_CELLS, img: int = IMG):
        super().__init__()
        M = torch.zeros(n_cells, img)
        for i, (a, b) in enumerate(cell_bins(n_cells, img)):
            M[i, a:b] = 1.0 / (b - a)
        self.register_buffer("M", M)

    def forward(self, x):
        return self.M @ x @ self.M.T


class Encoder(nn.Module):
    """Four conv stages, each followed by 2x2 max pooling (indices kept, as in SegNet)."""

    def __init__(self, cin: int = 1, base: int = 32):
        super().__init__()
        self.chs = [base, base * 2, base * 4, base * 8]
        ins = [cin] + self.chs[:-1]
        self.stages = nn.ModuleList(conv_block(i, o) for i, o in zip(ins, self.chs))
        self.bottleneck = conv_block(self.chs[-1], self.chs[-1])

    def forward(self, x):
        skips, idxs = [], []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
            x, idx = F.max_pool2d(x, 2, return_indices=True)
            idxs.append(idx)
        return self.bottleneck(x), skips, idxs


class SegNetBackbone(nn.Module):
    """Encoder-decoder: max-unpooling with stored indices (SegNet) + skip connections (paper Fig. 2b)."""

    def __init__(self, cin: int = 1, base: int = 32):
        super().__init__()
        self.enc = Encoder(cin, base)
        chs = self.enc.chs  # [b, 2b, 4b, 8b]
        outs = [chs[2], chs[1], chs[0], chs[0]]
        self.dec = nn.ModuleList(conv_block(2 * c, o) for c, o in zip(reversed(chs), outs))
        self.out_ch = chs[0]

    def forward(self, x):
        x, skips, idxs = self.enc(x)
        for blk, skip, idx in zip(self.dec, reversed(skips), reversed(idxs)):
            x = F.max_unpool2d(x, idx, 2, output_size=skip.shape[-2:])
            x = blk(torch.cat([x, skip], dim=1))
        return x


class SegMet(nn.Module):
    """M_theta: phase map (B,1,224,224) -> logits (B, 2, 31, 20, 20) for (dx, dy).

    Per-pixel class logits are produced at full resolution (segmentation) and
    averaged over each unit cell's footprint to get one prediction per cell.
    """

    def __init__(self, base: int = 32, n_classes: int = N_CLASSES, n_cells: int = N_CELLS):
        super().__init__()
        self.backbone = SegNetBackbone(1, base)
        self.head = nn.Conv2d(self.backbone.out_ch, 2 * n_classes, 1)
        self.pool = CellPool(n_cells)
        self.n_classes, self.n_cells = n_classes, n_cells

    def forward(self, x):
        z = self.pool(self.head(self.backbone(x)))
        return z.view(z.shape[0], 2, self.n_classes, self.n_cells, self.n_cells)


class SegNetRegression(nn.Module):
    """Ablation: identical encoder-decoder, but a 2-channel regression head (MSE)."""

    def __init__(self, base: int = 32, n_cells: int = N_CELLS):
        super().__init__()
        self.backbone = SegNetBackbone(1, base)
        self.head = nn.Conv2d(self.backbone.out_ch, 2, 1)
        self.pool = CellPool(n_cells)

    def forward(self, x):
        return self.pool(self.head(self.backbone(x)))


class CNNRegression(nn.Module):
    """Baseline in the spirit of Nadell et al. (2019): conv feature extractor + fully
    connected regression of all 2 x 20 x 20 continuous design parameters."""

    def __init__(self, base: int = 32, n_cells: int = N_CELLS, hidden: int = 512):
        super().__init__()
        self.enc = Encoder(1, base)
        self.reduce = nn.Sequential(nn.Conv2d(self.enc.chs[-1], 32, 1), nn.ReLU(inplace=True))
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, hidden), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(hidden, 2 * n_cells * n_cells),
        )
        self.n_cells = n_cells

    def forward(self, x):
        z, _, _ = self.enc(x)
        z = F.adaptive_avg_pool2d(self.reduce(z), 7)
        return self.fc(z).view(x.shape[0], 2, self.n_cells, self.n_cells)
