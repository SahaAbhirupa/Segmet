"""Synthetic near-field dataset for SegMet.

The paper builds its dataset with Lumerical FDTD. That is not available here,
so this module implements a cheap, deterministic, physics-inspired surrogate
that keeps the structure of the paper's problem:

* a 20 x 20 grid of unit cells (period P), each holding five elliptical
  Si3N4 pillars with semi-axes (r0 + dx, r0 + dy), dx, dy in {-15..15} nm;
* a random focusing lens phase from Eq. (1) of the paper (random wavelength,
  focal length, n_eff and lateral focus offset) on which the meta-atom
  response is superimposed;
* thin-element transmission  t = A(C) * exp(i [phi_lens + dphi_pillar * C]),
  where C is the anti-aliased pillar coverage rendered on a 4x supersampled
  grid (so 1 nm changes in a semi-axis produce a real, sub-pixel signal);
* angular-spectrum propagation (including evanescent decay) to a near-field
  monitor plane a distance z above the surface, which couples neighbouring
  cells and blurs sub-wavelength detail;
* detector integration down to 224 x 224 pixels plus complex Gaussian noise;
* the observed input is the wrapped phase angle(E) in [-pi, pi].

The sample is therefore (phi, dx, dy): a 224x224 single-channel phase image and
two 20x20 integer label maps, exactly the I/O of the paper's model M_theta.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr

N_CELLS = 20
IMG = 224
N_CLASSES = 31
DELTA_MAX = 15  # nm; classes 0..30  <->  delta -15..15 nm


@dataclass
class GenConfig:
    n_cells: int = N_CELLS
    img: int = IMG
    supersample: int = 4
    period_um: float = 1.0          # unit-cell period -> 20 um aperture (Fig. 1a)
    r0_um: float = 0.15             # nominal ellipse semi-axis
    atom_offset: float = 0.25       # four outer ellipses at (+-o, +-o) * P
    n_pillar: float = 2.02          # Si3N4
    h_range_um: tuple = (0.25, 0.35)    # effective pillar phase thickness
    amp_loss: float = 0.1           # amplitude drop over a pillar
    z_um: float = 0.05              # near-field monitor height
    wl_range_um: tuple = (0.45, 0.70)   # visible
    f_range_um: tuple = (5.0, 30.0)
    neff_range: tuple = (1.5, 2.1)
    focus_shift_um: float = 3.0     # random lateral focus offset (focused vs. shifted)
    noise: float = 0.02             # complex field noise std (field amplitude ~1)
    label_corr_range: tuple = (0.0, 0.8)   # weight of the smooth component of the label field
    label_len_range: tuple = (1.0, 4.0)    # correlation length of the smooth component (cells)


def delta_to_class(d):
    return np.asarray(d) + DELTA_MAX


def class_to_delta(c):
    return np.asarray(c) - DELTA_MAX


def cell_bins(n_cells: int = N_CELLS, img: int = IMG):
    """Pixel bins of each cell, identical to torch adaptive pooling bins."""
    return [(int(np.floor(i * img / n_cells)), int(np.ceil((i + 1) * img / n_cells)))
            for i in range(n_cells)]


def sample_labels(rng: np.random.Generator, cfg: GenConfig) -> np.ndarray:
    """Two 20x20 class maps with uniform marginals and random spatial correlation.

    A Gaussian field mixing a smooth component (lens-like slowly varying design)
    with an i.i.d. component is pushed through the normal CDF, giving uniform
    marginals over the 31 classes (majority-class accuracy ~ 1/31).
    """
    out = np.empty((2, cfg.n_cells, cfg.n_cells), dtype=np.int64)
    for k in range(2):
        rho = rng.uniform(*cfg.label_corr_range)
        ell = rng.uniform(*cfg.label_len_range)
        smooth = gaussian_filter(rng.standard_normal((cfg.n_cells,) * 2), ell, mode="wrap")
        smooth /= smooth.std() + 1e-12
        iid = rng.standard_normal((cfg.n_cells,) * 2)
        z = np.sqrt(rho) * smooth + np.sqrt(1 - rho) * iid
        out[k] = np.clip(np.floor(ndtr(z) * N_CLASSES), 0, N_CLASSES - 1)
    return out


def sample_params(rng: np.random.Generator, cfg: GenConfig) -> dict:
    return dict(
        wl=rng.uniform(*cfg.wl_range_um),
        f=rng.uniform(*cfg.f_range_um),
        neff=rng.uniform(*cfg.neff_range),
        h=rng.uniform(*cfg.h_range_um),
        x0=rng.uniform(-cfg.focus_shift_um, cfg.focus_shift_um),
        y0=rng.uniform(-cfg.focus_shift_um, cfg.focus_shift_um),
    )


class Simulator:
    """Forward model: (labels, lens params) -> near-field complex field / phase."""

    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        n = cfg.img * cfg.supersample
        self.n = n
        L = cfg.n_cells * cfg.period_um
        self.dx = L / n
        c = (np.arange(n) + 0.5) * self.dx
        self.X, self.Y = np.meshgrid(c, c)  # rows = y, cols = x
        P = cfg.period_um
        self.ci = np.minimum((self.Y // P).astype(np.int64), cfg.n_cells - 1)
        self.cj = np.minimum((self.X // P).astype(np.int64), cfg.n_cells - 1)
        self.u = self.X - (self.cj + 0.5) * P
        self.v = self.Y - (self.ci + 0.5) * P
        o = cfg.atom_offset * P
        self.offsets = [(0.0, 0.0), (o, o), (o, -o), (-o, o), (-o, -o)]
        fx = np.fft.fftfreq(n, d=self.dx)
        self.KX, self.KY = np.meshgrid(2 * np.pi * fx, 2 * np.pi * fx)
        self.Xc = self.X - L / 2
        self.Yc = self.Y - L / 2

    def coverage(self, labels: np.ndarray) -> np.ndarray:
        """Anti-aliased pillar coverage in [0, 1] on the supersampled grid."""
        cfg = self.cfg
        dx_nm, dy_nm = class_to_delta(labels[0]), class_to_delta(labels[1])
        a = cfg.r0_um + 1e-3 * dx_nm[self.ci, self.cj]
        b = cfg.r0_um + 1e-3 * dy_nm[self.ci, self.cj]
        cov = np.zeros_like(self.X)
        for ox, oy in self.offsets:
            p = (self.u - ox) / a
            q = (self.v - oy) / b
            rho = np.sqrt(p * p + q * q) + 1e-12
            grad = np.sqrt((p / a) ** 2 + (q / b) ** 2) / rho + 1e-12
            sd = (rho - 1.0) / grad  # approx. signed distance to the ellipse edge (um)
            cov = np.maximum(cov, np.clip(0.5 - sd / self.dx, 0.0, 1.0))
        return cov

    def lens_phase(self, prm: dict) -> np.ndarray:
        """Eq. (1): hyperbolic focusing phase with a lateral focus offset."""
        k = 2 * np.pi / prm["wl"]
        r2 = (self.Xc - prm["x0"]) ** 2 + (self.Yc - prm["y0"]) ** 2
        return -k * np.sqrt(r2 + prm["f"] ** 2) - prm["neff"] * k

    def field(self, labels: np.ndarray, prm: dict, rng: np.random.Generator | None = None,
              noise: float | None = None) -> np.ndarray:
        cfg = self.cfg
        noise = cfg.noise if noise is None else noise
        C = self.coverage(labels)
        k = 2 * np.pi / prm["wl"]
        dphi = k * (cfg.n_pillar - 1.0) * prm["h"]
        t = (1.0 - cfg.amp_loss * C) * np.exp(1j * (self.lens_phase(prm) + dphi * C))
        # angular-spectrum propagation; evanescent waves decay (kz imaginary)
        kz = np.sqrt((k ** 2 - self.KX ** 2 - self.KY ** 2).astype(np.complex128))
        E = np.fft.ifft2(np.fft.fft2(t) * np.exp(1j * kz * cfg.z_um))
        s = cfg.supersample
        E = E.reshape(cfg.img, s, cfg.img, s).mean(axis=(1, 3))  # detector pixels
        if noise > 0:
            assert rng is not None
            E = E + noise * (rng.standard_normal(E.shape) + 1j * rng.standard_normal(E.shape)) / np.sqrt(2)
        return E

    def phase(self, labels, prm, rng=None, noise=None) -> np.ndarray:
        return np.angle(self.field(labels, prm, rng, noise)).astype(np.float32)


def generate(n: int, seed: int, cfg: GenConfig | None = None) -> dict:
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(seed)
    sim = Simulator(cfg)
    X = np.empty((n, cfg.img, cfg.img), dtype=np.float32)
    Y = np.empty((n, 2, cfg.n_cells, cfg.n_cells), dtype=np.int64)
    params = []
    for i in range(n):
        Y[i] = sample_labels(rng, cfg)
        prm = sample_params(rng, cfg)
        X[i] = sim.phase(Y[i], prm, rng)
        params.append(prm)
    return dict(X=X, Y=Y, params=params, cfg=asdict(cfg), seed=seed)


def split(n: int, seed: int, frac=(0.6, 0.2, 0.2)):
    """60/20/20 split as in the paper."""
    idx = np.random.default_rng(seed + 10_000).permutation(n)
    n_tr = int(round(frac[0] * n))
    n_va = int(round(frac[1] * n))
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
