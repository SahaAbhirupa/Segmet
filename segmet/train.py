"""Training / evaluation of the deep models."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import DELTA_MAX
from .metrics import summarize
from .models import CNNRegression, SegMet, SegNetRegression

MODELS = {
    "segmet": (SegMet, "cls"),
    "cnn_reg": (CNNRegression, "reg"),
    "segnet_reg": (SegNetRegression, "reg"),
}


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def segmet_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Eq. (2): cross-entropy on dx plus cross-entropy on dy (averaged over cells)."""
    return (F.cross_entropy(logits[:, 0], y[:, 0]) + F.cross_entropy(logits[:, 1], y[:, 1]))


def predict_classes(model, kind: str, X: torch.Tensor, bs: int = 16) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            o = model(X[i:i + bs])
            if kind == "cls":
                c = o.argmax(dim=2)
            else:
                c = torch.round(o * DELTA_MAX).clamp(-DELTA_MAX, DELTA_MAX).long() + DELTA_MAX
            out.append(c.cpu())
    return torch.cat(out).numpy()


def train_model(name: str, data: dict, tr, va, te, seed: int, epochs: int = 150, bs: int = 4,
                lr: float = 1e-3, base: int = 32, augment: bool = True, log=print) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    cls, kind = MODELS[name]
    model = cls(base=base).to(dev)

    X = torch.from_numpy(data["X"] / np.pi).unsqueeze(1).to(dev)
    Y = torch.from_numpy(data["Y"]).to(dev)
    Yr = (Y - DELTA_MAX).float() / DELTA_MAX
    tr_t = torch.as_tensor(tr)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(tr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(seed)

    best, best_state, curve = -1.0, None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), generator=g)]
        tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs].to(dev)
            x, y, yr = X[b], Y[b], Yr[b]
            if augment:
                # mirror flips are symmetries of the forward model; label maps flip with the
                # image (a mirror does not swap the x/y semi-axes, unlike a 90-degree rotation)
                r = torch.rand(2, generator=g)
                dims = [d for d, flag in zip((-1, -2), r < 0.5) if flag]
                if dims:
                    x, y = x.flip(dims), y.flip(dims)
                    yr = yr.flip(dims)
            out = model(x)
            loss = segmet_loss(out, y) if kind == "cls" else F.mse_loss(out, yr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        va_acc = summarize(predict_classes(model, kind, X[va]), data["Y"][va])["acc"]
        curve.append(dict(epoch=ep + 1, train_loss=tot / len(tr), val_acc=va_acc))
        if va_acc > best:
            best, best_state = va_acc, copy.deepcopy(model.state_dict())
        if (ep + 1) % max(1, epochs // 10) == 0:
            log(f"  [{name}] ep {ep + 1:4d} loss {tot / len(tr):.4f} val_acc {va_acc:.4f} "
                f"best {best:.4f} ({time.time() - t0:.0f}s)")
    model.load_state_dict(best_state)
    pred = predict_classes(model, kind, X[te])
    res = summarize(pred, data["Y"][te])
    res.update(val_acc=best, curve=curve, train_time_s=time.time() - t0,
               n_params=sum(p.numel() for p in model.parameters()))
    res["_pred"] = pred
    return res
