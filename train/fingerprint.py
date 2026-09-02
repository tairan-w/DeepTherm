"""Learned molecular representation extraction."""

import torch


def fingerprint(model, batch):
    model.eval()
    with torch.no_grad():
        return model.encode(batch)


__all__ = ["fingerprint"]
