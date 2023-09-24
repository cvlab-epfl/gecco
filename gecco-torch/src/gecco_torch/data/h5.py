import os
import torch
import imageio as iio
import numpy as np
import h5py

from gecco.structs import Example, Context3d

class H5Scene:
    def __init__(
        self,
        h5_path: str,
        n_points: int = 2048,
    ):
        self.n_points = n_points
        self.h5_path = h5_path
        
        try:
            with h5py.File(h5_path, 'r') as hdf:
                self.fnames = [fname.decode('utf-8') for fname in hdf['fnames']]
        except Exception as e:
            raise IOError(f'Error opening {h5_path=}.') from e
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, index):
        image = self._get_image(index)
        pc, K = self._get_pc_and_K(index)

        return Example(
            data=pc,
            ctx=Context3d(
                image=image,
                K=K,
            ),
        )

    def __repr__(self):
        return (f'H5Scene(h5_path={self.h5_path}, '
                f'n_points={self.n_points}, '
                f'len={len(self)})')
    
    def _get_pc_and_K(self, index):
        with h5py.File(self.h5_path, 'r') as hdf:
            pc = torch.from_numpy(hdf['clouds'][index])
            K = torch.from_numpy(hdf['Ks'][index])
        
        subset = torch.randperm(pc.shape[0])[:self.n_points]
        pc = pc[subset].to(torch.float32)
        
        return pc, K
    
    def _get_image_path(self, index):
        return os.path.join(
            os.path.dirname(self.h5_path),
            self.fnames[index],
        )

    def _get_image(self, index):
        bitmap = iio.imread(self._get_image_path(index))
        bitmap = torch.from_numpy(bitmap)
        return bitmap.permute(2, 0, 1).float() / 255