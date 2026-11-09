"""Data loading for the generated project.

Falls back to synthetic tensors of the right shape whenever the real dataset is
unavailable, so the verification gates can run offline.
"""
import os
import warnings

import torch
from torch.utils.data import DataLoader, TensorDataset

TORCHVISION = {"CIFAR10", "CIFAR100", "MNIST", "FashionMNIST", "SVHN"}

# Tasks this loader can build batches for. The others are valid in a spec --
# a paper is allowed to be about segmentation -- but need their own loader,
# so say so instead of handing back classification data that looks fine.
CLASSIFICATION = {"image_classification", "text_classification",
                  "tabular_classification"}


def _synthetic(cfg, n, batch_size):
    shape = cfg["input_shape"]
    classes = cfg["output_dim"] or 2
    x = torch.randn(n, *shape)
    y = torch.randint(0, classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def _torchvision(cfg, batch_size, limit):
    import torchvision
    import torchvision.transforms as T

    name = cfg["dataset"]["name"]
    tf = T.Compose([T.ToTensor()])
    ctor = getattr(torchvision.datasets, name)
    train = ctor(root="./data", train=True, download=True, transform=tf)
    test = ctor(root="./data", train=False, download=True, transform=tf)
    if limit:
        train = torch.utils.data.Subset(train, range(min(limit, len(train))))
        test = torch.utils.data.Subset(test, range(min(limit, len(test))))
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(test, batch_size=batch_size))


def get_loaders(cfg, limit=None, batch_size=None):
    task = cfg["dataset"].get("task", "image_classification")
    if task not in CLASSIFICATION:
        raise NotImplementedError(
            f"this scaffold only builds classification batches; {task} needs "
            "a task-specific data.py")
    batch_size = batch_size or cfg["training"].get("batch_size") or 32
    name = cfg["dataset"]["name"]
    # The verification gates set this: they prove the code runs, and making
    # that proof wait on a dataset download would be slow and need a network.
    # A real training run leaves it unset and gets the real data.
    synthetic_only = os.environ.get("RMP_DATA") == "synthetic"
    if (not synthetic_only
            and cfg["dataset"].get("source") == "torchvision"
            and name in TORCHVISION):
        try:
            return _torchvision(cfg, batch_size, limit)
        except Exception as exc:  # network, disk, API drift
            warnings.warn(f"{name} unavailable ({exc}); using synthetic data")
    n = limit or 512
    return _synthetic(cfg, n, batch_size), _synthetic(cfg, max(n // 4, 32), batch_size)
