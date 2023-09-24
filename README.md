# GECCO
This is the official code release for the paper [GECCO: Geometrically-Conditioned Point Diffusion Models](https://arxiv.org/abs/2303.05916). It contains two implementations: using JAX and PyTorch. The experiments described in the paper were carried out with a JAX codebase. Unfortunately the JAX ecosystem is in a state of constant flux, meaning the package will only work with specific dependency versions, making it difficult to use for future work. For the convenience of the community we also provide a PyTorch reimplementation, as a starting point for future work. In short: if you want to reproduce the numbers (possibly on a different dataset): use JAX. If you want to modify/extend this work - use PyTorch.

## Usage
This section contains only infomation shared between the JAX and PyTorch version. For details see the README in each respective package.

### Installation
This repository is structured as two separate `pip` packages which can be installed via `pip install path/to/this/repository/gecco-{jax,torch}` and used as `import gecco_{jax,torch}`. Add the `-e` flag in your `pip install` command if you want your changes in this repository to be immediately reflected in import locations, otherwise you need to re-install the packages after each modification.

### Configuration
We find that with machine learning research projects a lot of time is spent updating a sophisticated configuration parser to cover all the functionality, much of which is soon to be abandoned as soon as the experiment is found to bring no improvements. Instead, this project simply uses `.py` files for configuration with the assumption that they define a `model` object and start training when executed.

### Data
#### ShapeNet-vol (image-conditional)
We use the data provided with the Occupancy Networks paper. The download script is [here](https://github.com/autonomousvision/occupancy_networks/blob/406f79468fb8b57b3e76816aaa73b1915c53ad22/scripts/download_data.sh) and will download and unpack the data.

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
We use the data provided with the [PointFlow](https://github.com/stevenygd/PointFlow#dataset) repository: download from [this](https://drive.google.com/drive/folders/1G0rf-6HSHoTll6aH7voh-dXj6hCRhSAQ?usp=sharing) and unpack the repository.

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