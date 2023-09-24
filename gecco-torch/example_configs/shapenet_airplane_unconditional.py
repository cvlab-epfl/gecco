import os

import torch
import lightning.pytorch as pl

from gecco_torch.diffusion import EDMPrecond, Diffusion, IdleConditioner
from gecco_torch.reparam import GaussianReparam
from gecco_torch.diffusion import Diffusion, LogUniformSchedule, EDMLoss
from gecco_torch.models.set_transformer import SetTransformer
from gecco_torch.models.lift3d import LinearLift
from gecco_torch.models.activation import GaussianActivation
from gecco_torch.data.shapenet_unc import ShapeNetUncondDataModule
from gecco_torch.ema import EMACallback
from gecco_torch.vis import PCVisCallback

dataset_path = '/cvlabdata2/cvlab/datasets_tyszkiewicz/point-flow-data/ShapeNetCore.v2.PC15k/'
data = ShapeNetUncondDataModule(
    dataset_path,
    category='airplane',
    epoch_size=5_000,
    batch_size=48,
    num_workers=16,
)

reparam = GaussianReparam(
    mean=torch.tensor([0.0, 0.01, 0.05]),
    sigma=torch.tensor([0.11, 0.04, 0.17]),
)

feature_dim = 3 * 128
network = LinearLift(
    inner=SetTransformer(
        n_layers=6,
        num_inducers=64,
        feature_dim=feature_dim,
        t_embed_dim=1,
        num_heads=8,
        activation=GaussianActivation,
    ),
    feature_dim=feature_dim,
)

model = Diffusion(
    backbone=EDMPrecond(
        model=network,
    ),
    conditioner=IdleConditioner(),
    reparam=reparam,
    loss=EDMLoss(
        schedule=LogUniformSchedule(
            max=165.0,
        ),
    ),
)

def trainer():
    return pl.Trainer(
        default_root_dir=os.path.split(__file__)[0],
        callbacks=[
            EMACallback(decay=0.99),
            pl.callbacks.ModelCheckpoint(),
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                filename='{epoch}-{val_loss:.3f}',
                save_top_k=1,
                mode='min',
            ),
            PCVisCallback(n=8, n_steps=128, point_size=0.01),
        ],
        max_epochs=50,
        precision='16-mixed',
        gradient_clip_val=1.0,
        gradient_clip_algorithm='value',
    )

if __name__ == '__main__':
    model = torch.compile(model)
    trainer().fit(model, data)