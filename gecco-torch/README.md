# GECCO
This is the official PyTorch reimplementation of the paper [GECCO: Geometrically-Conditioned Point Diffusion Models](https://arxiv.org/abs/2303.05916). The experiments described in the paper were carried out with a JAX codebase, which is also released. Unfortunately the JAX ecosystem is in a state of constant flux, meaning the package will only work with specific dependency version. For this reason we provide this PyTorch reimplementation for the convenience of the community, as a starting point for future work. If you want to reproduce our results most reliably, please use the JAX version instead.

## Usage
### Installation
This package can be installed with `pip` via `pip install path/to/this/repository` and used as `import gecco`. Use `pip install -e path/to/this/repository` if you want your changes in this repository to be immediately reflected in import locations, otherwise you need to re-install the package after each modification.

### Configuration
We find that with machine learning research projects a lot of time is spent updating a sophisticated configuration parser to cover all the functionality, much of which is soon to be abandoned as soon as the experiment is found to bring no improvements. Instead, this project simply uses `.py` files for configuration with the assumption that they define a `model` object and start training when executed. Please see `example_configs/{shapenet_airplane_unconditional.py,taskonomy_conditional.py}` for extensively commented examples.

### Data
#### ShapeNet-vol (image-conditional)
We use the data provided with the Occupancy Networks paper. The download script is [here](https://github.com/autonomousvision/occupancy_networks/blob/406f79468fb8b57b3e76816aaa73b1915c53ad22/scripts/download_data.sh) and will download and unpack the data. Afterwards the dataset is ready to use with `gecco.data.ShapeNetCondDataModule`.

The download folder should have the following structure:
```
ShapeNet/
├── 02691156
├── 02828884
├── 02933112
├── 02958343
├── 03001627
├── 03211117
├── 03636649
├── 03691459
├── 04090263
├── 04256520
├── 04379243
├── 04401088
├── 04530566
└── metadata.yaml

13 directories, 1 file
```
Your config files should point to the location of the root `ShapeNet` directory.

#### ShapeNet-PointFlow (unconditional)
We use the data provided with the [PointFlow](https://github.com/stevenygd/PointFlow#dataset) repository: download from [this](https://drive.google.com/drive/folders/1G0rf-6HSHoTll6aH7voh-dXj6hCRhSAQ?usp=sharing) and unpack the repository. Afterwards you can use it by pointing the `root` argument of `gecco.data.ShapeNetUncondDataModule` to the `ShapeNetCore.v2.PC15k` directory.

```
ShapeNetCore.v2.PC15k/
├── 02691156
├── 02747177
├── 02773838
├── 02801938
├── 04379243
|   ...
├── 04401088
├── 04460130
├── 04468005
├── 04530566
└── 04554684

55 directories, 0 files
```

#### Taskonomy (conditional)
Coming soon.

### Training
This codebase uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) to streamline training. To start, simply edit the example configs and adapt the data paths then execute the config file. If training on your own data, look at `notebooks/find_hyperparameters.ipynb` for an instruction on setting the reparametrization and noise schedule parameters.

### Inference
To prepare your model for inference, you should
1. Import the model definition from the config file.
2. Load model weights (be careful to use the exponential moving average weights, not the "regular" ones)
3. Use sampling methods on the model object

Since the config files may be in various places outside of this directory, we provide a utility called `gecco.load_config` to load the config files by path (instead of the standard import system). Your usecases should be mostly covered by the notebooks in `notebooks/` but we present a minimal example below:

```python
import torch
import gecco

config_root = 'path/to/experiment/directory'
config = gecco.load_config(f'{config_root}/config.py') # load the model definition
model = config.model
state_dict = torch.load(f'{config_root}/lightning_logs/version_0/checkpoints/last.ckpt', map_location='cpu')
model.load_state_dict(state_dict['ema_state_dict'])
model = model.eval()

samples = model.sample_stochastic(
    (1, 2048, 3), # one example with 2048 3-dimensional points
    context=None, # assuming an unconditional model
    with_pbar=True, # shows a tqdm progress bar for sampling
)
```
