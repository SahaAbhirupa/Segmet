"""Comparative study: Linear Regression / Linear Classification / CNN Regression / SegMet.

Each (experiment, setting, model, seed) result is cached as JSON under results/runs/,
so the study can be interrupted and resumed. Usage:

    python run_study.py --exp main            # paper scale: 100 samples, 60/20/20, 3 seeds
    python run_study.py --exp noise           # noise sweep at paper scale
    python run_study.py --exp scale           # 1000 samples
    python make_report.py                     # tables + figures
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from segmet import baselines
from segmet.data import GenConfig, generate, split
from segmet.train import device, train_model

RUNS = Path("results/runs")
ALL_MODELS = ["majority", "lin_reg", "lin_cls", "cnn_reg", "segnet_reg", "segmet"]

EXPERIMENTS = {
    # name: list of (setting tag, n_samples, noise, epochs, seeds)
    "main": [("n100_noise0.02", 100, 0.02, 150, [0, 1, 2])],
    "noise": [(f"n100_noise{s}", 100, s, 150, [0, 1, 2]) for s in (0.0, 0.05, 0.1)],
    "scale": [("n1000_noise0.02", 1000, 0.02, 60, [0])],
}


def run_one(model: str, data, tr, va, te, seed: int, epochs: int) -> dict:
    if model == "majority":
        return baselines.majority(data, tr, va, te)
    if model == "lin_reg":
        return baselines.linear_regression(data, tr, va, te)
    if model == "lin_cls":
        return baselines.linear_classification(data, tr, va, te)
    return train_model(model, data, tr, va, te, seed, epochs=epochs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=list(EXPERIMENTS), default="main")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    args = ap.parse_args()
    print(f"device: {device()}")

    for tag, n, noise, epochs, seeds in EXPERIMENTS[args.exp]:
        for seed in args.seeds or seeds:
            data = None
            for model in args.models:
                out = RUNS / tag / f"{model}_seed{seed}.json"
                if out.exists():
                    print(f"skip {out}")
                    continue
                if data is None:
                    t0 = time.time()
                    # the seed controls both the generated dataset and the training run
                    data = generate(n, seed, GenConfig(noise=noise))
                    tr, va, te = split(n, seed)
                    print(f"[{tag} seed {seed}] generated {n} samples in {time.time() - t0:.0f}s")
                t0 = time.time()
                res = run_one(model, data, tr, va, te, seed, epochs)
                res["wall_s"] = time.time() - t0
                pred = res.pop("_pred", None)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(dict(model=model, seed=seed, n=n, noise=noise,
                                               epochs=epochs, **res), indent=1))
                if pred is not None and seed == 0:
                    np.savez_compressed(out.with_suffix(".npz"), pred=pred, test_idx=te,
                                        true=data["Y"][te], X=data["X"][te[:4]])
                print(f"[{tag} seed {seed}] {model:10s} acc={res['acc']:.4f} "
                      f"pm1={res['acc_pm1']:.4f} mae={res['mae_nm']:.2f}nm ({res['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
