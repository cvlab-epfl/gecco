import os
import numpy as np
import torch
import lightning.pytorch as pl

from gecco_torch.structs import Example
from gecco_torch.data.samplers import ConcatenatedSampler, FixedSampler

id_to_name = {
'04379243': 'table',
'03593526': 'jar',
'04225987': 'skateboard',
'02958343': 'car',
'02876657': 'bottle',
'04460130': 'tower',
'03001627': 'chair',
'02871439': 'bookshelf',
'02942699': 'camera',
'02691156': 'airplane',
'03642806': 'laptop',
'02801938': 'basket',
'04256520': 'sofa',
'03624134': 'knife',
'02946921': 'can',
'04090263': 'rifle',
'04468005': 'train',
'03938244': 'pillow',
'03636649': 'lamp',
'02747177': 'trash bin',
'03710193': 'mailbox',
'04530566': 'watercraft',
'03790512': 'motorbike',
'03207941': 'dishwasher',
'02828884': 'bench',
'03948459': 'pistol',
'04099429': 'rocket',
'03691459': 'loudspeaker',
'03337140': 'file cabinet',
'02773838': 'bag',
'02933112': 'cabinet',
'02818832': 'bed',
'02843684': 'birdhouse',
'03211117': 'display',
'03928116': 'piano',
'03261776': 'earphone',
'04401088': 'telephone',
'04330267': 'stove',
'03759954': 'microphone',
'02924116': 'bus',
'03797390': 'mug',
'04074963': 'remote',
'02808440': 'bathtub',
'02880940': 'bowl',
'03085013': 'keyboard',
'03467517': 'guitar',
'04554684': 'washer',
'02834778': 'bicycle',
'03325088': 'faucet',
'04004475': 'printer',
'02954340': 'cap',
}

name_to_id = {v: k for k, v in id_to_name.items()}

class TorchShapenet:
    def __init__(
        self,
        root: str,
        category: str,
        split: str,
        n_points: int = 2048,
        batch_size: int = 48,
    ):
        self.path = os.path.join(root, category, split)
        if not os.path.exists(self.path):
            id = name_to_id[category]
            self.path = os.path.join(root, id, split)
        if not os.path.exists(self.path):
            raise ValueError(f'Path {self.path} does not exist')

        self.npys = [f for f in os.listdir(self.path) if f.endswith('.npy')]
        self.n_points = n_points
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.npys)
    
    def __getitem__(self, index):
        points = np.load(os.path.join(self.path, self.npys[index]))
        points = torch.from_numpy(points).to(torch.float32)
        perm = torch.randperm(points.shape[0])[:self.n_points]
        selected = points[perm].clone()
        return Example(selected, [])

class ShapeNetUncondDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        category: str,
        n_points: int = 2048,
        batch_size: int = 48,
        num_workers: int = 8,
        epoch_size: int | None = 10_000,
    ):
        super().__init__()

        self.root = root
        self.category = category
        self.n_points = n_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch_size = epoch_size

    def setup(self, stage=None):
        self.train = TorchShapenet(self.root, self.category, 'train', self.n_points)
        self.val = TorchShapenet(self.root, self.category, 'val', self.n_points)
        self.test = TorchShapenet(self.root, self.category, 'test', self.n_points)

    def train_dataloader(self):
        if self.epoch_size is None:
            kw = dict(
                shuffle=True,
                sampler=None,
            )
        else:
            kw = dict(
                shuffle=False,
                sampler=ConcatenatedSampler(self.train, self.epoch_size * self.batch_size, seed=None),
            )

        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kw,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=FixedSampler(self.val, length=None, seed=42),
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
