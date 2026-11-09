"""Verification gates, run one stage at a time by the pipeline."""
import argparse
import json

import torch
import torch.nn as nn
import yaml

from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["forward", "step"], required=True)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(0)
    model = build_model(cfg)
    x = torch.randn(2, *cfg["input_shape"])

    out = model(x)
    expected = cfg["output_dim"]
    if expected and tuple(out.shape) != (2, expected):
        raise SystemExit(
            f"output shape {tuple(out.shape)} != expected (2, {expected})"
        )

    if args.stage == "step":
        criterion = getattr(nn, cfg["training"].get("loss") or "CrossEntropyLoss")()
        loss = criterion(out, torch.randint(0, expected or 2, (2,)))
        loss.backward()
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss: {loss.item()}")
        grads = sum(1 for p in model.parameters() if p.grad is not None)
        if grads == 0:
            raise SystemExit("no parameter received a gradient")

    print(json.dumps({
        "stage": args.stage,
        "output_shape": list(out.shape),
        "params": sum(p.numel() for p in model.parameters()),
    }))


if __name__ == "__main__":
    main()
