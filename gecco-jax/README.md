# GECCO (JAX)
This is the official JAX implementation of the paper [GECCO: Geometrically-Conditioned Point Diffusion Models](https://arxiv.org/abs/2303.05916). Unfortunately the JAX ecosystem is in a state of constant flux, meaning this package will only work with specific dependency version (see `pyproject.toml`). For this reason we provide also provide PyTorch reimplementation for the convenience of the community, as a starting point for future work. This codebase is provided mostly for reliable reproduction of our results.

## Usage
### Installation
This package can be installed with `pip` via `pip install path/to/this/repository` and used as `import gecco_jax`. Use `pip install -e path/to/this/repository` if you want your changes in this repository to be immediately reflected in import locations, otherwise you need to re-install the package after each modification.

### Configuration
We find that with machine learning research projects a lot of time is spent updating a sophisticated configuration parser to cover all the functionality, much of which is soon to be abandoned as soon as the experiment is found to bring no improvements. Instead, this project simply uses `.py` files for configuration with the assumption that they define a `model` object and start training when executed. Please see `example_configs/{shapenet_airplane_unconditional.py,taskonomy_conditional.py}` for extensively commented examples.

### Data
See the top level README in this repository.

### Training
TODO

### Inference
To prepare your model for inference, you should
1. Import the model definition from the config file.
2. Load model weights (be careful to use the exponential moving average weights, not the "regular" ones)
3. Use sampling methods on the model object

Since the config files may be in various places outside of this directory, we provide a utility called `gecco.load_config` to load the config files by path (instead of the standard import system). Your usecases should be mostly covered by the notebooks in `notebooks/` but we present a minimal example below:

TODO