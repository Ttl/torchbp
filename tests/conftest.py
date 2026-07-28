"""Shared test configuration and helpers.

Importable from the test modules as ``from conftest import ...``: pytest puts
this directory on ``sys.path`` (the test package has no ``__init__.py``), and
so does running a test file directly.
"""
import unittest

import torch

#: Speed of light in vacuum [m/s]. Must match ``kC0`` in ``csrc/cpu/util.h``
#: and :data:`torchbp.autofocus.C0`.
C0 = 299792458.0

#: Skip decorator for tests that need a CUDA device.
requires_cuda = unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
