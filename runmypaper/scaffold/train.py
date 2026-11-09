"""Training entry point for the generated project."""
import argparse
import json

import torch
import torch.nn as nn
import yaml

from data import get_loaders
from model import build_model


def build_optimizer(cfg, params):
    t = cfg["training"]
    kwargs = {"lr": t.get("lr") or 0.01}
    if t.get("weight_decay") is not None:
        kwargs["weight_decay"] = t["weight_decay"]
    name = (t.get("optimizer") or "SGD").strip()
    if name.lower() == "sgd" and t.get("momentum") is not None:
        kwargs["momentum"] = t["momentum"]
    return getattr(torch.optim, name)(params, **kwargs)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit", type=int, help="cap samples per split (smoke runs)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = get_loaders(cfg, limit=args.limit)
    model = build_model(cfg).to(device)
    opt = build_optimizer(cfg, model.parameters())
    criterion = getattr(nn, cfg["training"].get("loss") or "CrossEntropyLoss")()
    epochs = args.epochs or cfg["training"].get("epochs") or 1

    history = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            opt.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            running += loss.item()
        history.append(running / max(len(train_loader), 1))

    print(json.dumps({
        "epochs": epochs,
        "loss_history": history,
        "accuracy": evaluate(model, test_loader, device),
        "params": sum(p.numel() for p in model.parameters()),
    }, indent=2))


if __name__ == "__main__":
    main()
