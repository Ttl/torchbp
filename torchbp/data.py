"""Lazy out-of-core data sources for backprojection functions.

A :class:`LazyData` stands in for the range compressed ``data`` tensor when
the raw data does not fit in memory. It mimics the small part of the Tensor
interface that the library uses (``shape``, ``dtype``, ``device``, ``dim()``,
``len()`` and contiguous slicing along the sweep axis) so that a Tensor and a
LazyData are interchangeable at every Python call site; the data is
materialized with :meth:`LazyData.load` only where a kernel consumes it.

Slicing (``lazy[i0:i1]``) is free: it returns a view restricted to the sweep
range without loading anything. :func:`torchbp.ops.ffbp` slices subaperture
views down its recursion tree and loads only leaf-sized chunks, so its peak
memory stays at image scale; :func:`torchbp.autofocus.gpga` and
:func:`torchbp.autofocus.gpga_tde` additionally read the data in bounded
chunks in their estimation stage. Functions that need the full tensor at once
(direct backprojection, afbp, cfbp) accept a LazyData but load the whole
extent at entry, so they are not memory efficient with it.

Gradients do not flow to the data through a LazyData (``requires_grad`` is
always False). Gradients with respect to positions are unaffected.
"""

import os
import sys
import tempfile

import numpy as np
import torch
from torch import Tensor

__all__ = ["LazyData", "CallbackData", "MemmapData", "CachedData",
           "materialize", "available_ram"]


def available_ram() -> int:
    """Available system RAM in bytes, counting reclaimable caches as
    available.

    Use it to size a :class:`CachedData` RAM/disk decision, e.g.
    ``storage="ram" if nbytes < available_ram() // 2 else "disk"``.

    - Linux: ``MemAvailable`` from ``/proc/meminfo``, the kernel's estimate
      of memory usable without swapping; it counts reclaimable page cache
      and slab, unlike ``MemFree``.
    - Windows: ``GlobalMemoryStatusEx`` ``ullAvailPhys``, which includes
      the standby (file cache) list — the same semantics.
    - Elsewhere (or on failure): half of total physical RAM as a
      conservative estimate, or 2 GiB if even that is unavailable.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    if sys.platform == "win32":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullAvailPhys)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2
    except (ValueError, OSError, AttributeError):
        return 2 << 30


def _disk_tmp_dir() -> str:
    """Directory for disk cache scratch files.

    The system temp dir, unless it is RAM-backed (a tmpfs /tmp is common
    on Linux), in which case a RAM-saving "disk" cache would silently
    consume the RAM it is supposed to save. In that case /var/tmp, which the
    FHS guarantees to be persistent (disk-backed).
    """
    d = tempfile.gettempdir()
    try:
        fstype = None
        best = -1
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt = parts[1]
                if (d == mnt or mnt == "/"
                        or d.startswith(mnt.rstrip("/") + "/")):
                    if len(mnt) > best:
                        fstype = parts[2]
                        best = len(mnt)
        if fstype in ("tmpfs", "ramfs") and os.path.isdir("/var/tmp"):
            return "/var/tmp"
    except OSError:
        pass
    return d


class LazyData:
    """Base class for lazy radar data sources.

    Subclasses implement :meth:`_load_range` returning the sweeps
    ``[start, stop)`` as a Tensor (or anything ``torch.as_tensor`` accepts)
    and call ``super().__init__`` with the full extent ``shape``, the
    ``dtype``/``device`` the loaded chunks should have, and an optional
    ``transform``.

    Parameters
    ----------
    shape : tuple
        Shape of the full data, ``[nsweeps, samples]``.
    dtype : torch.dtype
        Dtype of the loaded data (after conversion), typically
        ``torch.complex64``.
    device : str or torch.device
        Device the loaded chunks are moved to. This is the device the
        backprojection runs on; ``lazy.device`` reports it so existing
        ``device = data.device`` call sites keep working.
    transform : callable or None
        Optional per-chunk pipeline ``transform(chunk, start, stop) ->
        Tensor`` applied after the chunk is moved to ``device`` but before
        dtype conversion. Use it for preprocessing that would otherwise
        force materializing the full tensor up front: windowing, range
        compression of raw sweeps, RVP removal, modulation.
        ``start``/``stop`` are the absolute sweep indices of the chunk, so
        per-sweep factors (an azimuth window over the full aperture) can be
        indexed. The transform may change the dtype and the number of
        samples per sweep — ``shape`` and ``dtype`` describe its output —
        but must not change the number of sweeps. For an iterative consumer
        (:func:`torchbp.autofocus.gpga`) that re-reads the data every
        iteration, wrap the source in :class:`CachedData` so an expensive
        pipeline runs only once per sweep.
    """

    def __init__(self, shape: tuple, dtype: torch.dtype, device,
                 transform=None):
        self._shape = torch.Size(shape)
        self._dtype = dtype
        self._device = torch.device(device)
        self._transform = transform

    # Tensor-compatible attribute surface used by the library.
    @property
    def shape(self) -> torch.Size:
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def requires_grad(self) -> bool:
        return False

    def dim(self) -> int:
        return len(self._shape)

    def __len__(self) -> int:
        return self._shape[0]

    def resolve_conj(self) -> "LazyData":
        # Loaded chunks are freshly materialized tensors, never lazy
        # conjugate views.
        return self

    def _load_range(self, start: int, stop: int):
        raise NotImplementedError

    def _load(self, start: int, stop: int) -> Tensor:
        n = stop - start
        if n <= 0:
            return torch.empty((0,) + tuple(self._shape[1:]),
                               dtype=self._dtype, device=self._device)
        t = self._load_range(start, stop)
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t)
        # Device first, then transform, then dtype: the transform receives
        # the raw chunk on the compute device and may change its dtype and
        # sample count (e.g. range compression of raw real ADC sweeps), so
        # the declared shape/dtype describe its output, not its input.
        if t.device != self._device:
            t = t.to(self._device, non_blocking=True)
        if self._transform is not None:
            # A transform ending in .conj() returns a lazy conjugate view;
            # resolve it so consumers never see the conj bit (the dispatcher
            # would silently materialize it again on every kernel call).
            t = self._transform(t, start, stop).resolve_conj()
        if t.dtype != self._dtype:
            t = t.to(self._dtype)
        expected = (n,) + tuple(self._shape[1:])
        if tuple(t.shape) != expected:
            raise RuntimeError(
                f"{type(self).__name__} loaded shape {tuple(t.shape)} for "
                f"sweeps [{start}, {stop}), expected {expected}")
        return t

    def load(self) -> Tensor:
        """Materialize the full extent of this source (or view) as a Tensor."""
        return self._load(0, len(self))

    def __getitem__(self, item) -> "LazyData":
        if not isinstance(item, slice):
            raise TypeError(
                "LazyData supports only contiguous slicing along the sweep "
                "axis (data[i0:i1]); call load() to materialize a Tensor for "
                "anything else")
        start, stop, step = item.indices(len(self))
        if step != 1:
            raise TypeError("LazyData slicing does not support a step")
        return _LazyDataView(self, start, max(start, stop))


class _LazyDataView(LazyData):
    """Zero-cost view of a LazyData restricted to a sweep range.

    Nested views stay flat: slicing a view returns another view into the
    root source with absolute indices.
    """

    def __init__(self, root: LazyData, start: int, stop: int):
        super().__init__((stop - start,) + tuple(root.shape[1:]),
                         root.dtype, root.device)
        self._root = root
        self._start = start

    def _load(self, start: int, stop: int) -> Tensor:
        return self._root._load(self._start + start, self._start + stop)

    def __getitem__(self, item) -> "LazyData":
        if not isinstance(item, slice):
            raise TypeError(
                "LazyData supports only contiguous slicing along the sweep "
                "axis (data[i0:i1]); call load() to materialize a Tensor for "
                "anything else")
        start, stop, step = item.indices(len(self))
        if step != 1:
            raise TypeError("LazyData slicing does not support a step")
        return _LazyDataView(self._root, self._start + start,
                             self._start + max(start, stop))


class CallbackData(LazyData):
    """Lazy data source backed by a user callback.

    Parameters
    ----------
    load_fn : callable
        ``load_fn(start, stop) -> Tensor`` returning the range compressed
        sweeps ``[start, stop)``, shape ``[stop - start, samples]``. May
        return anything ``torch.as_tensor`` accepts; the result is converted
        to ``dtype`` and moved to ``device``.
    shape : tuple
        Full data shape ``[nsweeps, samples]``.
    dtype : torch.dtype
        Dtype of the data. Default is ``torch.complex64``.
    device : str or torch.device
        Device the backprojection runs on. Default is ``"cpu"``.
    transform : callable or None
        Optional per-chunk hook, see :class:`LazyData`.
    """

    def __init__(self, load_fn, shape: tuple,
                 dtype: torch.dtype = torch.complex64, device="cpu",
                 transform=None):
        super().__init__(shape, dtype, device, transform)
        self._load_fn = load_fn

    def _load_range(self, start: int, stop: int):
        return self._load_fn(start, stop)


class MemmapData(LazyData):
    """Lazy data source backed by a sliceable array on disk.

    Wraps any array supporting ``array[start:stop]``, ``.shape`` and
    ``.dtype`` — ``numpy.memmap``, an h5py dataset, a zarr array — and loads
    slices on demand.

    Parameters
    ----------
    array : array-like
        Sliceable array of shape ``[nsweeps, samples]``.
    device : str or torch.device
        Device the backprojection runs on. Default is ``"cpu"``.
    dtype : torch.dtype or None
        Dtype of the loaded data after the transform. Default None keeps
        the array's own dtype; pass it explicitly when the transform
        changes the dtype.
    transform : callable or None
        Optional per-chunk pipeline, see :class:`LazyData`.
    shape : tuple or None
        Output shape override, ``[nsweeps, samples]``. Default None uses
        the array's shape; pass it when the transform changes the number
        of samples per sweep.
    """

    def __init__(self, array, device="cpu", dtype: torch.dtype | None = None,
                 transform=None, shape: tuple | None = None):
        if dtype is None:
            dtype = torch.as_tensor(array[0:0]).dtype
        if shape is None:
            shape = tuple(array.shape)
        super().__init__(shape, dtype, device, transform)
        self._array = array

    def _load_range(self, start: int, stop: int):
        a = self._array[start:stop]
        if isinstance(a, np.ndarray) and not a.flags.writeable:
            # torch.as_tensor warns on read-only arrays (e.g. a mmap_mode="r"
            # numpy array); the chunk is transformed/copied downstream anyway.
            a = a.copy()
        return torch.as_tensor(a)


class CachedData(LazyData):
    """Cache of another :class:`LazyData`'s transformed output.

    The first load of a sweep runs the source's pipeline and writes the
    result to the cache; later loads of the same sweeps read the cache
    instead. Use it when an expensive per-chunk transform (range
    compression, RVP removal) feeds an iterative consumer like
    :func:`torchbp.autofocus.gpga`, which re-reads the data every
    iteration: the pipeline then runs once per sweep instead of once per
    iteration.

    Only 2D ``[nsweeps, samples]`` sources are supported. The cache is
    keyed per sweep, so overlapping or out-of-order loads fill it
    incrementally. Call :meth:`fill` to populate it eagerly and release
    the source, so the raw data backing the pipeline can be freed before
    image formation.

    Parameters
    ----------
    source : LazyData
        The source (typically with a ``transform`` pipeline) to cache.
    path : str or None
        Cache file path for ``storage="disk"``. Default None creates a
        temporary file that is deleted when this object is garbage
        collected. A given path is left on disk (but always created anew,
        never trusted stale). The default avoids a RAM-backed system temp
        dir (tmpfs /tmp) by falling back to ``/var/tmp``.
    storage : str
        - "disk" (default): memory-mapped scratch file. Bounded RAM use;
          for data larger than memory.
        - "ram": a tensor on ``storage_device``. For data that fits, a
          filled RAM cache on the compute device is equivalent to an
          eagerly transformed tensor, loads are zero-copy views, so the
          lazy plumbing adds no per-load overhead, while :meth:`fill`
          still lets the raw source be freed.
    storage_device : str, torch.device or None
        Device of the "ram" cache tensor. Default None uses the source's
        (compute) device. Pass "cpu" with a CUDA compute device when the
        data fits in system RAM but not in VRAM: the cache then stages in
        pinned CPU memory and loads copy to the GPU chunk by chunk. Not
        used with ``storage="disk"``.
    """

    def __init__(self, source: LazyData, path: str | None = None,
                 storage: str = "disk", storage_device=None):
        if source.dim() != 2:
            raise ValueError("CachedData supports only 2D [nsweeps, "
                             f"samples] sources, got shape {tuple(source.shape)}")
        super().__init__(tuple(source.shape), source.dtype, source.device)
        self._source = source
        self._cached = np.zeros(source.shape[0], dtype=bool)
        if storage == "ram":
            if path is not None:
                raise ValueError("path is only used with storage='disk'")
            self._cache = None
            cache_dev = torch.device(
                source.device if storage_device is None else storage_device)
            # A CPU cache staging for a CUDA compute device is pinned so
            # the per-load host-to-device copies run at full bandwidth.
            pin = cache_dev.type == "cpu" and self._device.type == "cuda"
            self._cache_t = torch.empty(
                tuple(source.shape), dtype=source.dtype, device=cache_dev,
                pin_memory=pin)
        elif storage == "disk":
            np_dtype = torch.zeros((), dtype=source.dtype).numpy().dtype
            if path is None:
                self._tmp = tempfile.NamedTemporaryFile(
                    suffix=".torchbp-cache", dir=_disk_tmp_dir())
                path = self._tmp.name
            self._cache = np.memmap(path, dtype=np_dtype, mode="w+",
                                    shape=tuple(source.shape))
            self._cache_t = None
        else:
            raise ValueError(
                f"storage must be 'disk' or 'ram', got {storage!r}")

    def _fill_range(self, start: int, stop: int) -> None:
        missing = ~self._cached[start:stop]
        if not missing.any():
            return
        if self._source is None:
            raise RuntimeError(
                "CachedData source was released by fill() but sweeps "
                f"[{start}, {stop}) are not fully cached")
        # Fill contiguous missing runs through the source pipeline.
        idx = missing.nonzero()[0] + start
        runs = []
        r0 = prev = int(idx[0])
        for i in idx[1:]:
            i = int(i)
            if i != prev + 1:
                runs.append((r0, prev + 1))
                r0 = i
            prev = i
        runs.append((r0, prev + 1))
        for a, b in runs:
            chunk = self._source._load(a, b)
            if self._cache_t is not None:
                self._cache_t[a:b] = chunk.to(self._cache_t.device)
            else:
                self._cache[a:b] = chunk.cpu().numpy()
            self._cached[a:b] = True

    def fill(self, block_bytes: int = 64 << 20) -> "CachedData":
        """Eagerly populate the whole cache in bounded blocks and release
        the source.

        After this the cache is the only backing store: the source (and
        whatever raw data its pipeline reads) is dropped and can be freed
        by the caller, restoring the memory timeline of an eager range
        compression, compress once, free the raw data, with the
        compressed data in the cache storage instead of pinned by the
        pipeline. Returns self for chaining.
        """
        n = self.shape[0]
        row_bytes = torch.empty(0, dtype=self._dtype).element_size() * int(
            np.prod(self._shape[1:], dtype=np.int64))
        block = max(1, block_bytes // max(1, row_bytes))
        for a in range(0, n, block):
            self._fill_range(a, min(a + block, n))
        self._source = None
        return self

    def _load(self, start: int, stop: int) -> Tensor:
        if stop <= start:
            return super()._load(start, stop)
        self._fill_range(start, stop)
        if self._cache_t is not None:
            t = self._cache_t[start:stop]
        else:
            t = torch.from_numpy(np.ascontiguousarray(self._cache[start:stop]))
        if t.device != self._device:
            t = t.to(self._device, non_blocking=True)
        return t


def materialize(data) -> Tensor:
    """Return ``data`` as a Tensor, loading it if it is a :class:`LazyData`."""
    if isinstance(data, LazyData):
        return data.load()
    return data
