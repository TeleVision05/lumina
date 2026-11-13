"""Train LuminaNet on synthetic crops generated on the fly (Apple MPS, CPU fallback).

    uv run python -m lumina_train.train --run main --steps 60000

Every step renders a fresh batch: GROUPS panoramas from the training split, CROPS_PER_GROUP random
camera frames each (camera.py, isp.py, synth.py). JPEG and the flip run on a 4-thread CPU pool
while the GPU trains on the previous batch. The fixed validation set (fixedsets.py) is scored with
the EMA weights every --val-every steps. Logs (JSON lines) and checkpoints go to data/runs/<run>/.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor, nn

from lumina_train import metrics
from lumina_train.fixedsets import load_or_build_val_set
from lumina_train.hdri import load_split_cache
from lumina_train.losses import LightingLoss
from lumina_train.model import LuminaNet, count_parameters
from lumina_train.params import sample_params, select
from lumina_train.paths import RUNS_DIR
from lumina_train.synth import PanoBank, finish_images

GROUPS = 32
CROPS_PER_GROUP = 8


@dataclass
class Config:
    run: str
    steps: int
    lr: float
    weight_decay: float
    warmup: int
    val_every: int
    ema: float
    seed: int


def pick_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class BatchStream:
    """Renders batch n+1 on the device while batch n trains; JPEG runs on CPU threads."""

    def __init__(self, bank: PanoBank, seed: int) -> None:
        self.bank = bank
        self.rng = np.random.default_rng(seed)
        self.pool = ThreadPoolExecutor(max_workers=4)

    def submit(self) -> tuple[list[Future[Tensor]], Tensor]:
        """Start a batch: (futures of 4 image chunks on the CPU, labels on the device)."""
        n = GROUPS * CROPS_PER_GROUP
        idx = self.rng.choice(len(self.bank), GROUPS, replace=False)
        params = sample_params(self.rng, n)
        with torch.no_grad():
            noise = torch.randn(n, 3, 96, 128, device=self.bank.device)
            images, labels, _ = self.bank.render_device(idx, params, CROPS_PER_GROUP, noise)
            host = images.cpu()  # waits for the render; the JPEG then overlaps training
        bounds = np.linspace(0, n, 5).astype(int)
        futures = [
            self.pool.submit(finish_images, host[a:b], select(params, slice(a, b)))
            for a, b in itertools.pairwise(bounds)
        ]
        return futures, labels


class Ema:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.model = LuminaNet().to(next(model.parameters()).device)
        self.model.load_state_dict(model.state_dict())
        self.model.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for e, p in zip(self.model.state_dict().values(), model.state_dict().values(), strict=True):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
            else:
                e.copy_(p)


def lr_at(step: int, cfg: Config) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    t = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0))))


def make_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr)


@torch.no_grad()
def validate(model: nn.Module, val: dict[str, Tensor], loss_fn: LightingLoss) -> dict[str, float]:
    preds = []
    for start in range(0, len(val["images"]), 256):
        preds.append(model(val["images"][start : start + 256]))
    pred = torch.cat(preds)
    parts = loss_fn(pred, val["labels"])
    out = {f"val_{k}": float(v) for k, v in parts.parts.items()}
    out["val_loss"] = float(parts.total)
    per = metrics.compute(pred, val["labels"])
    for key in metrics.METRICS:
        out[f"val_{key}"] = float(np.mean(per[key]))
    strong = per["directionality"] >= metrics.DIRECTIONAL_THRESHOLD
    out["val_dir_deg_directional"] = float(np.mean(per["dir_deg"][strong]))
    return out


def load_val(device: torch.device) -> dict[str, Tensor]:
    raw = load_or_build_val_set()
    images = torch.from_numpy(raw["images"]).permute(0, 3, 1, 2).float() / 255.0
    return {"images": images.to(device), "labels": torch.from_numpy(raw["labels"]).to(device)}


def save_checkpoint(
    path: Path, model: nn.Module, ema: Ema, opt: Any, step: int, cfg: Config
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "config": asdict(cfg),
        },
        path,
    )


def train(cfg: Config, resume: bool) -> Path:
    dev = pick_device()
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(4)  # the owner's laptop: keep CPU use to 4 threads plus the JPEG pool
    cv2.setNumThreads(1)
    run_dir = RUNS_DIR / cfg.run
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = load_split_cache("train")
    bank = PanoBank(cache["panos"], cache["sh"], dev)
    del cache
    val = load_val(dev)
    model = LuminaNet().to(dev)
    ema = Ema(model, cfg.ema)
    opt = make_optimizer(model, cfg)
    loss_fn = LightingLoss(dev)
    start = 0
    if resume and (run_dir / "last.pt").exists():
        ckpt = torch.load(run_dir / "last.pt", map_location=dev)
        model.load_state_dict(ckpt["model"])
        ema.model.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["optimizer"])
        start = int(ckpt["step"])
    print(f"device {dev}, {count_parameters(model)} parameters, start step {start}", flush=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1) + "\n")
    stream = BatchStream(bank, cfg.seed * 1000 + start)
    _loop(cfg, start, model, ema, opt, loss_fn, stream, val, run_dir)
    return run_dir


def _best_so_far(run_dir: Path) -> float:
    path = run_dir / "log.jsonl"
    if not path.exists():
        return math.inf
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return min((r["val_loss"] for r in rows), default=math.inf)


def _loop(
    cfg: Config,
    start: int,
    model: LuminaNet,
    ema: Ema,
    opt: torch.optim.Optimizer,
    loss_fn: LightingLoss,
    stream: BatchStream,
    val: dict[str, Tensor],
    run_dir: Path,
) -> None:
    best = _best_so_far(run_dir)
    log = (run_dir / "log.jsonl").open("a")
    pending = stream.submit()
    t0, seen, sums = time.time(), 0, {}
    for step in range(start, cfg.steps):
        futures, labels = pending
        pending = stream.submit() if step + 1 < cfg.steps else pending
        images = torch.cat([f.result() for f in futures]).to(labels.device)
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)
        model.train()
        out = loss_fn(model(images), labels)
        opt.zero_grad(set_to_none=True)
        out.total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        for k, v in out.parts.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach())
        seen += 1
        if (step + 1) % cfg.val_every == 0 or step + 1 == cfg.steps:
            row: dict[str, Any] = {"step": step + 1, "lr": lr_at(step, cfg)}
            row |= {f"train_{k}": v / seen for k, v in sums.items()}
            row |= validate(ema.model, val, loss_fn)
            row["elapsed_s"] = time.time() - t0
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps({k: round(v, 4) for k, v in row.items()}), flush=True)
            save_checkpoint(run_dir / "last.pt", model, ema, opt, step + 1, cfg)
            if row["val_loss"] < best:
                best = row["val_loss"]
                save_checkpoint(run_dir / "best.pt", model, ema, opt, step + 1, cfg)
            seen, sums = 0, {}
    log.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="main")
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cfg = Config(
        run=args.run,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup=args.warmup,
        val_every=args.val_every,
        ema=args.ema,
        seed=args.seed,
    )
    train(cfg, args.resume)


if __name__ == "__main__":
    main()
