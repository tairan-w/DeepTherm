
from torch.utils.data import DataLoader

from .dataset import collate_graphs


def build_dataloader(dataset, batch_size=128, num_workers=0, shuffle=True, **kwargs):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_graphs,
        **kwargs,
    )


__all__ = ["build_dataloader"]
