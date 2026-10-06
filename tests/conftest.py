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


def range_compress(raw, oversample=2):
    """Range compress raw FMCW sweeps as the examples do.

    Hamming window, ``oversample`` times zero-padded IFFT and a modulation
    that centers the range spectrum. Returns ``(data, data_fmod)``; pass
    ``data_fmod`` to the image formation call.
    """
    n = raw.shape[-1]
    w = torch.hamming_window(n, periodic=False, device=raw.device)
    data = torch.fft.ifft(raw * w[None, :], dim=-1, n=n * oversample)
    data_fmod = -torch.pi * (1 - (oversample - 1) / oversample)
    k = torch.arange(data.shape[-1], device=raw.device)
    return data * torch.exp(1j * data_fmod * k)[None, :], data_fmod


def fmcw_scene(targets, pos, fstart, bw, tsweep, fs, oversample=2, wa=None):
    """Simulated range-compressed FMCW data of point targets.

    ``fstart`` is the frequency at the first sample of the sweep, the phase
    reference of the returned data; the spectral center is
    ``fstart + bw / 2``. Returns ``(data, data_fmod, r_res)``.
    """
    import torchbp

    rcs = torch.ones(targets.shape[0], device=targets.device)
    raw = torchbp.util.generate_fmcw_data(
        targets, rcs, pos, fstart, bw, tsweep, fs, rvp=False
    )
    if wa is not None:
        raw = raw * wa[:, None]
    data, data_fmod = range_compress(raw, oversample)
    return data, data_fmod, C0 / (2 * bw * oversample)
