from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader
from torch_dimcheck import A


@dataclass
class ConcatenatedSampler:
    data_length: int
    length: int
    rng: torch.Generator

    def __init__(
        self,
        data_source: Sequence,
        length: int,
        seed: int = 42,
    ):
        self.data_length = len(data_source)
        self.length = length
        self.seed = seed

    def __len__(self):
        return self.length

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed)
        yielded = 0

        while yielded < len(self):
            permutation = torch.randperm(self.data_length, generator=rng).numpy()
            left_to_yield = len(self) - yielded
            yield from permutation[:left_to_yield]
            yielded += permutation.shape[0]


@dataclass
class FixedSampler:
    permutation: A["N"]

    def __init__(
        self,
        dataset,
        length: Optional[int] = None,
        seed: int = 42,
    ):
        if length is None:
            length = len(dataset)
        if length > len(dataset):
            raise ValueError(f"{length=} is more than {len(dataset)=}.")

        rng: torch.Generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(dataset), generator=rng)
        self.permutation = perm[:length].numpy()

    def __len__(self):
        return self.permutation.shape[0]

    def __iter__(self):
        yield from self.permutation


def dataloader(
    dataset,
    batch_size: int,
    num_steps: Optional[int] = None,
    num_workers: int = 16,
    fixed_sampler: bool = False,
    sequential_sampler: bool = False,
    drop_last: Optional[bool] = None,
):
    if sequential_sampler and not fixed_sampler:
        raise AssertionError()

    if fixed_sampler:
        if sequential_sampler:
            sampler = "sequential"
        else:
            sampler = "fixed_perm"
    else:
        sampler = "concatenated"

    length = None if num_steps is None else batch_size * num_steps
    if sampler == "fixed_perm":
        kw = dict(
            sampler=FixedSampler(dataset, length=length),
            drop_last=False if drop_last is None else drop_last,
        )
    elif sampler == "sequential":
        kw = dict(
            shuffle=False,
            drop_last=False if drop_last is None else drop_last,
        )
    elif sampler == "concatenated":
        kw = dict(
            sampler=ConcatenatedSampler(dataset, length=length),
            drop_last=True if drop_last is None else drop_last,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        **kw,
    )
