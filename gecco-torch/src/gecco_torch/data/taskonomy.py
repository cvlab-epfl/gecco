import os
import re
from importlib import resources
from typing import Tuple, List

import torch
import numpy as np
import h5py
import lightning.pytorch as pl
from tqdm.auto import tqdm
import imageio as iio

from gecco.structs import Context3d, Example
from gecco.data.samplers import FixedSampler

class Building:
    def __init__(
        self,
        name: str,
        h5_path: str,
        rgb_path: str,
        n_points: int = 2048,
    ):
        self.name = name
        self.h5_path = os.path.join(h5_path, f'{name}.h5')
        self.rgb_path = os.path.join(rgb_path, name)
        self.n_points = n_points
    
        with h5py.File(self.h5_path, 'r') as h5_file:
            points = h5_file['point'][()]
            views = h5_file['view'][()]
        
        self.points_and_views = list(zip(points.tolist(), views.tolist()))
        missing_points_and_views = self.missing_points_and_views()
        is_available = ~np.array([(pv in missing_points_and_views) for pv in self.points_and_views])
        indices = np.arange(len(self.points_and_views))
        self.reindex = indices[is_available]

    def rgb_file_path(self, index, name_only: bool = False) -> str:
        point, view = self.points_and_views[index]
        fname = f'{self.name}_{point}_{view}.jpg'
        if name_only:
            return fname
        return os.path.join(self.rgb_path, fname)
    
    def missing_points_and_views(self) -> List[Tuple[int, int]]:
        existing_files = frozenset(os.listdir(self.rgb_path))
        requested_files = frozenset(
            self.rgb_file_path(i, name_only=True) for i in range(len(self.points_and_views))
        )
        
        missing_files = requested_files - existing_files
        
        fname_re = re.compile(r'\w+_(\d+)_(\d+)\.jpg')
        missing_points_and_views = set()
        for missing_file in missing_files:
            if (m := fname_re.match(missing_file)) is None:
                raise RuntimeError(f"{missing_file=} doesn't match format.")
            point = int(m.group(1))
            view = int(m.group(2))
            missing_points_and_views.add((point, view))

        return missing_points_and_views
    
    def __len__(self):
        return len(self.reindex)
    
    def __getitem__(self, index):
        index = self.reindex[index]

        with h5py.File(self.h5_path, 'r') as h5_file:
            pc = h5_file['pc'][index]
            K = h5_file['k'][index]

        image_path = self.rgb_file_path(index)
        image = np.asarray(iio.imread(image_path))
        image = image.astype(np.float32).transpose(2, 0, 1) / 255

        perm = np.random.permutation(pc.shape[0])[:self.n_points]
        pc = pc[perm]

        return Example(
            data=pc.astype(np.float32),
            ctx=Context3d(
                image=image,
                K=K,
            ),
        )

def parse_split_file(split_file):
    splits = {}
    for line in list(split_file)[1:]: # skip header
        name, is_train, is_val, is_test = line.split(',')
        if int(is_train):
            splits[name] = 'train'
        if int(is_val):
            splits[name] = 'val'
        if int(is_test):
            splits[name] = 'test'
    return splits

class Taskonomy(torch.utils.data.ConcatDataset):
    def __init__(self, path: str, split: str = 'all', n_points: int = 2048):
        self.h5_path = os.path.join(path, 'point_clouds')
        self.rgb_path = os.path.join(path, 'rgb')
        self.split = split

        with resources.open_text(
            'gecco.data',
            'taskonomy_split.csv',
        ) as split_file:
            splits = parse_split_file(split_file)
        
        if split == 'all':
            belongs_in_split = lambda _name: True
        else:
            belongs_in_split = lambda name: splits[name] == split

        buildings = []
        for file in tqdm(os.listdir(self.h5_path)):
            name = file[:-len('.h5')]

            if not belongs_in_split(name):
                continue

            buildings.append(Building(name, self.h5_path, self.rgb_path, n_points=n_points))
            
        super().__init__(buildings)
    
    def __repr__(self):
        return f'Taskonomy(split={self.split}, n_buildings={len(self.datasets)}, len={len(self)})'
    
class TaskonomyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        n_points: int = 2048,
        batch_size: int = 48,
        num_workers: int = 8,
        epoch_size: int | None = 10_000,
        val_size: int | None = 10_000,
    ):
        super().__init__()

        self.root = root
        self.n_points = n_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch_size = epoch_size
        self.val_size = val_size
    
    def setup(self, stage='fit'):
        if stage == 'fit':
            self.train = Taskonomy(self.root, 'train', self.n_points)
            self.val = Taskonomy(self.root, 'val', self.n_points)
        elif stage == 'test':
            self.test = Taskonomy(self.root, 'test', self.n_points)
        else:
            raise ValueError(f'Unknown stage {stage}')
        
    def train_dataloader(self):
        if self.epoch_size is None:
            kw = dict(
                shuffle=True,
                sampler=None,
                batch_size=self.batch_size,
            )
        else:
            kw = dict(
                shuffle=False,
                sampler=torch.utils.data.RandomSampler(self.train, replacement=True, num_samples=self.epoch_size),
                batch_size=self.batch_size,
            )
        return torch.utils.data.DataLoader(
            self.train,
            num_workers=self.num_workers,
            **kw,
        )
    
    def val_dataloader(self):
        if self.val_size is None:
            sampler = None
        else:
            sampler = FixedSampler(self.val, length=self.val_size)

        return torch.utils.data.DataLoader(
            self.val,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            sampler=sampler,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )