# Torchbp

![SAR image](https://github.com/Ttl/torchbp/blob/master/docs/img/07_19_1_autofocus_gamma0_pauli_pol_cal.png?raw=true)

Fast C++ Pytorch extension for differentiable synthetic aperture radar image formation and autofocus library on GPU.

Only Nvidia GPUs are supported. Currently, only some of the operations are supported on CPU.

On RTX 3090 Ti backprojection on polar grid achieves 225 billion
backprojections/s and fast factorized backprojection (ffbp) is ten times faster on
moderate size images.

## Installation

Tested with CUDA version 12.1, some newer versions might cause build issues.

### From source

```bash
git clone https://github.com/Ttl/torchbp.git
cd torchbp
pip install .
```

## Documentation

API documentation and examples can be built with sphinx.

```bash
pip install .[docs]
cd docs
make html
```

Open `docs/build/html/index.html`.
