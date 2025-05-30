# Torchbp

Fast C++ Pytorch extension for differentiable synthetic aperture radar image formation and autofocus library on CPU and GPU.

Only Nvidia GPUs are supported. Currently, some operations are not supported on CPU.

On RTX 3090 Ti backprojection on polar grid achieves 225 billion backprojections/s.

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
