# Torchbp

![SAR image](https://github.com/Ttl/torchbp/blob/master/docs/img/07_19_1_autofocus_sigma0_pol_cal_pauli.png?raw=true)

Fast C++ Pytorch extension for differentiable synthetic aperture radar image formation and autofocus library on GPU.

Only Nvidia GPUs are supported. Currently, only some of the operations are supported on CPU.

On RTX 3090 Ti backprojection on polar grid achieves 370 billion
backprojections/s and fast factorized backprojection (ffbp) is ten times faster on
moderate size images.

## Installation

Tested with CUDA version 12.9.

### From source

Install PyTorch first, then build torchbp against that same torch with
build isolation disabled:

```bash
pip install torch
git clone https://github.com/Ttl/torchbp.git
cd torchbp
pip install --no-build-isolation -e .
```

`--no-build-isolation` is required. The extension links libtorch C++ symbols
that are not ABI-stable across torch versions, so it must be compiled against
the exact torch you run at import time. Without the flag, pip builds in an
isolated environment with a different (freshly downloaded) torch, producing
a `.so` that fails to import with an `undefined symbol` error. For the same
reason, rebuild torchbp whenever you change your torch version.

## Documentation

Latest documentation can be viewed at: https://ttl.github.io/torchbp/

API documentation and examples can be built with sphinx.

```bash
pip install --no-build-isolation .[docs]
cd docs
make html
```

Open `docs/build/html/index.html`.
