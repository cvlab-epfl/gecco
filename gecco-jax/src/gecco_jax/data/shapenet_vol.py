import os
import re
from typing import Callable, NamedTuple, Optional, Union, List, Any
from functools import partial

import torch
import numpy as np
import imageio as iio
import multiprocess as mp
from tqdm.auto import tqdm

from gecco_jax.types import Example, Context3d, named_tuple_repr

IM_SIZE = 137 # 137 x 137 pixels
WORLD_MAT_RE = re.compile(r'world_mat_(\d+)')
CAMERA_MAT_RE = re.compile(r'camera_mat_(\d+)')
FIX_MASK_RE = re.compile(r'mask_(\d+)')
    
def _mask_npz_key(view: int) -> str:
    return f'mask_{view}'

class TestData(NamedTuple):
    points_raw: np.ndarray
    scale: np.ndarray
    loc: np.ndarray
    wmat: np.ndarray
    category: List[str]
    object_id: List[str]

    __repr__ = named_tuple_repr
    
class ShapeNetVolModel:
    def __init__(
        self,
        root: str,
        posed: bool = False,
        image_conditional: bool = False,
        n_points: int = 2048,
        skip_fixed: bool = False,
        is_testing: bool = False,
    ):
        if image_conditional and not posed:
            raise AssertionError('image_conditional=True is valid only with posed=True')
        
        self.root = root
        self.posed = posed
        self.image_conditional = image_conditional
        self.n_points = n_points
        self.skip_fixed = skip_fixed
        self.is_testing = is_testing
        
        self.wmats, self.cmats = None, None
        self._fixed_view_ids = None
        self._is_fixed = None

    @property
    def fixed_path(self) -> str:
        return os.path.join(self.root, 'per_view_point_masks.npz')
    
    @property
    def is_fixed(self):
        if self._is_fixed is None:
            self._is_fixed = os.path.exists(self.fixed_path)
        return self._is_fixed
        
    def get_camera_params(self, index: int):
        if self.wmats is None:
            npz = np.load(os.path.join(self.root, 'img_choy2016', 'cameras.npz'))
            
            world_mat_ids = set()
            camera_mat_ids = set()
            
            for key in npz.keys():
                if (m := WORLD_MAT_RE.match(key)) is not None:
                    world_mat_ids.add(int(m.group(1)))
                    continue
                if (m := CAMERA_MAT_RE.match(key)) is not None:
                    camera_mat_ids.add(int(m.group(1)))
                    continue
            
            assert world_mat_ids == camera_mat_ids
            
            indices = np.array(sorted(list(world_mat_ids)))
            if (indices != np.arange(24)).all():
                raise AssertionError('Bad shapenet model')

            world_mats = np.stack([npz[f'world_mat_{i}'] for i in indices])
            camera_mats = np.stack([npz[f'camera_mat_{i}'] for i in indices])

            # normalize camera matrices
            camera_mats /= np.array([IM_SIZE + 1, IM_SIZE + 1, 1]).reshape(3, 1)

            self.wmats = world_mats.astype(np.float32)
            self.cmats = camera_mats.astype(np.float32)
        
        return self.wmats[index], self.cmats[index]

    def get_fix_mask(self, view: int) -> Optional[np.ndarray]:
        # no fixes
        if not self.is_fixed:
            return None

        # cache exists and `view` isn't in there
        if (self._fixed_view_ids is not None) and (view not in self._fixed_view_ids):
            return None

        try:
            fix_file = np.load(self.fixed_path)
        except FileNotFoundError:
            return None
        
        # if the cache doesn't exist, build it
        if self._fixed_view_ids is None:
            fixed_view_ids = set()
            for key in fix_file.keys():
                if (m := FIX_MASK_RE.match(key)) is not None:
                    fixed_view_ids.add(int(m.group(1)))
            self._fixed_view_ids = frozenset(fixed_view_ids)

            # since we just built the cache, we may not be in there
            if view not in self._fixed_view_ids:
                return None
        
        return fix_file[_mask_npz_key(view)]

    @property
    def pointcloud_npz_path(self):
        return os.path.join(self.root, 'pointcloud.npz')

    def points_scale_loc(self):
        pc = np.load(self.pointcloud_npz_path)
        points = pc['points'].astype(np.float32)
        scale = pc['scale'].astype(np.float32)
        loc = pc['loc'].astype(np.float32)

        return points, scale, loc

    def points_world(self, view: Optional[int] = None):
        points, scale, loc = self.points_scale_loc()
        if view is not None:
            fix_mask = self.get_fix_mask(view)
            if fix_mask is not None:
                points = points[fix_mask]

        if self.n_points is not None:
            subset = np.random.permutation(points.shape[0])
            points = points[subset[:self.n_points]]
        return points * scale + loc[None, :]

    def __len__(self):
        if self.skip_fixed and self.is_fixed:
            return 0
        if self.is_testing:
            return 1
        if self.posed:
            return 24
        return 1
    
    def __getitem__(self, index) -> Example:
        if not self.posed:
            return Example(
                points=self.points_world(view=None),
            )

        wmat, cmat = self.get_camera_params(index)
        points = self.points_world(view=index)
        points_t = np.einsum('ab,nb->na', wmat[:, :3], points) + wmat[:, -1]

        if self.is_testing:
            *_prefix, category, object_id = self.root.split('/')
            points_raw, scale, loc = self.points_scale_loc()
            extras = TestData(
                wmat=wmat,
                points_raw=points_raw,
                scale=scale,
                loc=loc,
                category=category,
                object_id=object_id,
            )
        else:
            extras = ()
        
        if not self.image_conditional:
            return Example(
                points=points_t,
                ctx=Context3d(
                    image=[],
                    K=cmat,
                ),
                extras=extras,
            )

        image_index = index
        image = []
        image_path = os.path.join(
            self.root,
            'img_choy2016',
            f'{image_index:03d}.jpg',
        )
        image = iio.imread(image_path).astype(np.float32) / 255
        image = np.asarray(image)
        if image.ndim == 2: # grayscale to rgb
            image = image[..., None].repeat(3, 2)

        return Example(
            points=points_t,
            ctx=Context3d(
                image=image,
                K=cmat.copy(), # to avoid accidental mutation of self.cmats
                wmat=wmat.copy(),
            ),
            extras=extras,
        )

class ShapeNetVolClass(torch.utils.data.ConcatDataset):
    def __init__(
        self,
        root: str,
        split: str,
        **kw,
    ):
        with open(os.path.join(root, f'{split}.lst')) as split_file:
            split_ids = [line.strip() for line in split_file]
        paths = [os.path.join(root, id) for id in split_ids]
        make_model = partial(ShapeNetVolModel, **kw)
        
        if kw.get('posed', False) or kw.get('skip_fixed', False):
            # takes a while to load npzs to it's good to parallelize
            with mp.Pool() as pool:
                subsets = list(pool.imap(make_model, paths))
        else:
            # faster to not pay multiprocess overhead
            subsets = list(map(make_model, paths))

        super().__init__(subsets)
        self.root = root
        self.split = split

class ShapeNetVol(torch.utils.data.ConcatDataset):
    def __init__(
        self,
        root: str,
        split: Union[str, List[str]],
        transform: Callable[[Example], Example] = lambda e: e,
        **kw,
    ):
        if isinstance(split, str):
            subroots = []
            for maybe_dir in os.listdir(root):
                maybe_dir_path = os.path.join(root, maybe_dir)
                if not os.path.isdir(maybe_dir_path):
                    continue
                subroots.append(maybe_dir_path)
                
            super().__init__([
                ShapeNetVolClass(subroot, split, **kw) for subroot in tqdm(subroots)
            ])
        else:
            assert isinstance(split, (list, tuple))
            assert all(isinstance(path, str) for path in split)

            super().__init__([
                ShapeNetVolModel(path, **kw) for path in tqdm(split)
            ])

        self.transform = transform
    
    def __getitem__(self, index: int) -> Example:
        e = super().__getitem__(index)

        return self.transform(e)