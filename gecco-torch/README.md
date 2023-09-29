# GECCO (PyTorch)
This is the official PyTorch reimplementation of the paper [GECCO: Geometrically-Conditioned Point Diffusion Models](https://arxiv.org/abs/2303.05916). The experiments described in the paper were carried out with a JAX codebase, which is also released. Unfortunately the JAX ecosystem is in a state of constant flux, meaning the package will only work with specific dependency version. For this reason we provide this PyTorch reimplementation for the convenience of the community, as a starting point for future work. If you want to reproduce our results most reliably, please use the JAX version instead.

## Usage
### Installation
This package can be installed with `pip` via `pip install path/to/this/repository` and used as `import gecco_torch`. Use `pip install -e path/to/this/repository` if you want your changes in this repository to be immediately reflected in import locations, otherwise you need to re-install the package after each modification.

### Pretrained checkpoints
We provide pretrained checkpoints for both PyTorch and JAX [here](https://datasets.epfl.ch/gecco-weights/index.html).

> Note: due to insufficiently long training, the PyTorch checkpoints currently lag in performance between the JAX ones. They are still fine for qualitative evaluation and future work.

### Configuration
We find that with machine learning research projects a lot of time is spent updating a sophisticated configuration parser to cover all the functionality, much of which is soon to be abandoned as soon as the experiment is found to bring no improvements. Instead, this project simply uses `.py` files for configuration with the assumption that they define a `model` object and start training when executed. Please see `example_configs/{shapenet_airplane_unconditional.py,taskonomy_conditional.py}` for extensively commented examples.

### Data
See the top level README in this repository.

### Training
This codebase uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) to streamline training. To start, simply edit the example configs and adapt the data paths then execute the config file. If training on your own data, look at `notebooks/find_hyperparameters.ipynb` for an instruction on setting the reparametrization and noise schedule parameters.

### Inference
To prepare your model for inference, you should
1. Import the model definition from the config file.
2. Load model weights (be careful to use the exponential moving average weights, not the "regular" ones)
3. Use sampling methods on the model object

Since the config files may be in various places outside of this directory, we provide a utility called `gecco_torch.load_config` to load the config files by path (instead of the standard import system). Your usecases should be mostly covered by the notebooks in `notebooks/` but we present a minimal example below:

```python
import torch
import gecco_torch

config_root = 'path/to/experiment/directory'
config = gecco_torch.load_config(f'{config_root}/config.py') # load the model definition
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

### Known issues
1. This code does not implement all features of the JAX version:
    * No ODE sampler
    * No log-likelihood computation
    * No validation on benchmarks like the JAX codebase does
2. Training in 16bit precision eventually diverges. It is much faster (4-8x) than full 32bit precision so until the issue is addressed, I suggest training in 16bit until divergence and then resuming in 32 bit precision.
    * As a result the PyTorch checkpoints are under-trained. We will release improved versions soon.