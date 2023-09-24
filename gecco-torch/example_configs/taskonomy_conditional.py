import os

import torch
import lightning.pytorch as pl

from gecco_torch.diffusion import EDMPrecond, Diffusion
from gecco_torch.reparam import UVLReparam
from gecco_torch.diffusion import Diffusion, LogUniformSchedule, EDMLoss
from gecco_torch.models.set_transformer import SetTransformer
from gecco_torch.models.feature_pyramid import ConvNeXtExtractor
from gecco_torch.models.ray import RayNetwork
from gecco_torch.models.activation import GaussianActivation
from gecco_torch.vis import PCVisCallback
from gecco_torch.data.taskonomy import TaskonomyDataModule
from gecco_torch.ema import EMACallback

dataset_path = '/cvlabdata1/cvlab/datasets_tyszkiew/taskonomy_256x256/'
NUM_STEPS = 1_000_000
SAVE_EVERY = 10_000
BATCH_SIZE = 48
epoch_size = SAVE_EVERY * BATCH_SIZE
num_epochs = NUM_STEPS // SAVE_EVERY
print(f'num steps: {NUM_STEPS}, batch size: {BATCH_SIZE}, save_every: {SAVE_EVERY}, epoch size: {epoch_size}, num epochs: {num_epochs}')

# Reparametrization makes the point cloud more isotropic.
# The values below are computed with the notebook in notebooks/compute_hyperparams.ipynb
reparam = UVLReparam(
    uvl_mean=torch.tensor([0.0, 0.0, 1.38]),
    uvl_std=torch.tensor([0.56, 0.60, 0.49]),
)

# Set up the network. RayNetwork is responsible for augmenting cloud features with local
# features extracted from the context 2d image.
network = RayNetwork(
    backbone=SetTransformer( # a point cloud network with extra "global" input for the `t` parameter
        n_layers=6,
        num_inducers=64,
        feature_dim=3*128,
        t_embed_dim=1, # dimensionality of the `t` parameter
        num_heads=8,
        activation=GaussianActivation,
    ),
    reparam=reparam, # we need the reparam object to go between data and diffusion spaces for ray lookup
    context_dims=(96, 192, 384), # feature dimensions at scale 1, 2, 4
)

# Set up the diffusion model. This is largely agnostic of the
# 3d point cloud task and could be used for 2d image diffusion as well.
model = Diffusion(
    # We use EDMPrecond to precondition the diffusion as described in
    # https://arxiv.org/abs/2206.00364 (Table 1).
    backbone=EDMPrecond(
        model=network,
    ),
    
    # We use a ConvNeXtExtractor to extract features from the context image.
    # It is ran on Example.ctx of each batch and returns image feature pyramid.
    conditioner=ConvNeXtExtractor(),
    
    # Pass the diffusion reparametrization
    reparam=reparam,
    
    # We use the EDM loss as described in https://arxiv.org/abs/2206.00364
    # with a log-uniform (instead of log-Gaussian) noise schedule.
    loss=EDMLoss(
        schedule=LogUniformSchedule(
            max=180.0,
        ),
    ),
)

# set up the data module
data = TaskonomyDataModule(
    dataset_path,
    batch_size=BATCH_SIZE,
    num_workers=16,
    epoch_size=epoch_size,
    val_size=25*BATCH_SIZE,
)

def trainer():
    # it's convenient to have a function that returns a trainer
    # because it prints stuff to console at creation, so it's annoying
    # when using the config for inference/visualization. At the same time
    # we may want to access it outside of training for its callbacks etc
    return pl.Trainer(
        default_root_dir=os.path.split(__file__)[0],
        callbacks=[
            EMACallback(decay=0.999), # keep an exponential moving average of weights
            pl.callbacks.ModelCheckpoint(),
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                filename='{epoch}-{val_loss:.3f}',
                save_top_k=1,
                mode='min',
            ),
            PCVisCallback(n=8, n_steps=128, point_size=0.05), # visualize point clouds in tensorboard
        ],
        max_epochs=num_epochs,
        precision='16-mixed',
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
    )

if __name__ == '__main__':
    trainer().fit(model, data)