"""Aggregate results/runs/* into markdown tables (results/tables.md, and README.md
between the RESULTS markers) and draw the figures in results/figures/."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from segmet.data import DELTA_MAX

RUNS = Path("results/runs")
FIG = Path("results/figures")

NAMES = {
    "majority": "Majority class (chance)",
    "lin_reg": "`Linear Regression`",
    "lin_cls": "`Linear Classification`",
    "cnn_reg": "`CNN Regression`",
    "segnet_reg": "SegNet + regression head (ablation)",
    "segmet": "**`SegMet` (ours)**",
}
PAPER = {"lin_reg": "10.5%", "lin_cls": "18.3%", "cnn_reg": "22.7%", "segmet": "**67.8%**"}
ORDER = list(NAMES)

# reference palette (dataviz skill): categorical slots in fixed order, recessive ink
SERIES = {"segmet": "#2a78d6", "cnn_reg": "#eb6834", "segnet_reg": "#1baf7a",
          "lin_cls": "#eda100", "lin_reg": "#e87ba4", "majority": "#898781"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("bgo", ["#184f95", "#2a78d6", "#f0efec", "#eb6834", "#a8401a"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def load():
    res = defaultdict(lambda: defaultdict(list))  # tag -> model -> [run]
    for f in sorted(RUNS.glob("*/*.json")):
        r = json.loads(f.read_text())
        res[f.parent.name][r["model"]].append(r)
    return res


def ms(runs, key, pct=True):
    v = np.array([r[key] for r in runs]) * (100 if pct else 1)
    s = f"{v.mean():.1f} ± {v.std():.1f}" if len(v) > 1 else f"{v.mean():.1f}"
    return s + ("%" if pct else "")


def main_table(res, tag):
    rows = ["| Model | Paper | Accuracy (synthetic) | Acc. Δx | Acc. Δy | Within ±1 nm | MAE (nm) |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for m in ORDER:
        runs = res[tag].get(m)
        if not runs:
            continue
        rows.append(f"| {NAMES[m]} | {PAPER.get(m, '–')} | {ms(runs, 'acc')} | {ms(runs, 'acc_x')} | "
                    f"{ms(runs, 'acc_y')} | {ms(runs, 'acc_pm1')} | {ms(runs, 'mae_nm', False)} |")
    n = len(next(iter(res[tag].values())))
    return "\n".join(rows), n


def readme_table(res, tag):
    """Short table for the README: the paper's four methods only."""
    rows = ["| Model | Accuracy (synthetic data) |", "|---|---:|"]
    for m, name in [("lin_reg", "Linear Regression"), ("lin_cls", "Linear Classification"),
                    ("cnn_reg", "CNN Regression"), ("segmet", "**SegMet**")]:
        if res[tag].get(m):
            rows.append(f"| {name} | {ms(res[tag][m], 'acc')} |")
    return "\n".join(rows) + "\n"


def sweep_table(res, tags, header):
    rows = ["| Model | " + " | ".join(header) + " |", "|---|" + "---:|" * len(tags)]
    for m in ORDER:
        if not any(res[t].get(m) for t in tags):
            continue
        cells = [ms(res[t][m], "acc") if res[t].get(m) else "–" for t in tags]
        rows.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def fig_curves(res, tag):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m in ["segmet", "segnet_reg", "cnn_reg"]:
        runs = [r for r in res[tag].get(m, []) if r["seed"] == 0]
        if not runs:
            continue
        c = runs[0]["curve"]
        ep = [p["epoch"] for p in c]
        acc = [100 * p["val_acc"] for p in c]
        ax.plot(ep, acc, color=SERIES[m], lw=2, label=NAMES[m].strip("*`").replace("` (ours)", " (ours)"))
        ax.annotate(f"{acc[-1]:.1f}%", (ep[-1], acc[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
    ax.axhline(100 / 31, color=MUTED, lw=1, ls="--")
    ax.text(150 * 0.45, 100 / 31 + 0.35, "chance (1/31)", color=MUTED, fontsize=8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("Validation accuracy (seed 0, 100-sample synthetic set)", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.42))
    fig.tight_layout()
    fig.savefig(FIG / "val_accuracy.png", dpi=150)
    plt.close(fig)


def fig_noise(res, tags, noises):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m in ["segmet", "segnet_reg", "cnn_reg", "lin_cls", "lin_reg", "majority"]:
        pts = [(nz, 100 * np.mean([r["acc"] for r in res[t][m]])) for t, nz in zip(tags, noises) if res[t].get(m)]
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, color=SERIES[m], lw=2, marker="o", ms=5,
                label=NAMES[m].strip("*`").replace("` (ours)", " (ours)"))
    ax.set_xlabel("complex near-field noise σ")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Accuracy vs. simulation noise (100 samples, mean of seeds)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "noise_sweep.png", dpi=150)
    plt.close(fig)


def fig_sample(tag):
    f = RUNS / tag / "segmet_seed0.npz"
    if not f.exists():
        return
    d = np.load(f)
    true, pred, X = d["true"][0] - DELTA_MAX, d["pred"][0] - DELTA_MAX, d["X"][0]
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.2), gridspec_kw=dict(wspace=0.35))
    im = axs[0].imshow(X, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axs[0].set_title("input near-field phase φ")
    cb = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.03)
    cb.set_label("rad")
    panels = [(true[0], "ground-truth Δx"), (pred[0], "SegMet Δx"),
              (true[1], "ground-truth Δy"), (pred[1], "SegMet Δy")]
    for ax, (m, t) in zip(axs[1:], panels):
        im = ax.imshow(m, cmap=DIVERGING, vmin=-DELTA_MAX, vmax=DELTA_MAX)
        ax.set_title(t)
    cb = fig.colorbar(im, ax=axs[1:], fraction=0.012, pad=0.02)
    cb.set_label("nm")
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.savefig(FIG / "sample_prediction.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    res = load()
    parts = []
    if "n100_noise0.02" in res:
        t, n = main_table(res, "n100_noise0.02")
        parts.append(f"### Table 1 — paper scale: 100 samples (60/20/20), noise σ = 0.02, mean ± std over {n} seeds\n\n{t}")
        fig_curves(res, "n100_noise0.02")
        fig_sample("n100_noise0.02")
    noises = [0.0, 0.02, 0.05, 0.1]
    ntags = [f"n100_noise{s}" for s in noises]
    if sum(t in res for t in ntags) > 1:
        tags = [t for t in ntags if t in res]
        parts.append("### Table 2 — noise sweep (100 samples, accuracy, mean ± std over seeds)\n\n"
                     + sweep_table(res, tags, [f"σ = {t.split('noise')[1]}" for t in tags]))
        fig_noise(res, tags, [float(t.split("noise")[1]) for t in tags])
    if "n1000_noise0.02" in res:
        tags = ["n100_noise0.02", "n1000_noise0.02"]
        parts.append("### Table 3 — dataset size (noise σ = 0.02, accuracy, mean ± std over seeds)\n\n"
                     + sweep_table(res, tags, ["100 samples (60 train)", "1000 samples (600 train)"]))
    md = "\n\n".join(parts) + "\n"
    Path("results/tables.md").write_text(md)
    readme = Path("README.md")
    if readme.exists() and "n100_noise0.02" in res:
        table = readme_table(res, "n100_noise0.02")
        s = readme.read_text()
        s = re.sub(r"(<!-- RESULTS:START -->\n).*?(<!-- RESULTS:END -->)", lambda mt: mt.group(1) + table + mt.group(2),
                   s, flags=re.S)
        readme.write_text(s)
    print(md)


if __name__ == "__main__":
    main()
