import os
import glob

from setuptools import find_packages, setup

try:
    import torch
    from packaging.version import Version
    from torch.utils.cpp_extension import (
        CppExtension,
        CUDAExtension,
        BuildExtension,
        CUDA_HOME,
    )
except ImportError as e:
    raise RuntimeError(
        "Building torchbp requires PyTorch to be installed in the build "
        "environment. The extension links non-ABI-stable libtorch symbols, so "
        "it must be compiled against the *same* torch you run at import time.\n"
        "\n"
        "Install torch first, then build with build isolation disabled so the "
        "compile uses your installed torch instead of a freshly downloaded one:\n"
        "\n"
        "    pip install torch\n"
        "    pip install --no-build-isolation .\n"
    ) from e

library_name = "torchbp"

# Build against CPython's stable ABI (abi3) when torch supports it (>= 2.6).
# This gives a single, version-stable extension filename (_C.abi3.so) that is
# reused across CPython versions and overwritten in place on rebuild, instead
# of a per-Python name (_C.cpython-3XX-...so) that can leave stale duplicates
# behind. NOTE: this only stabilizes the Python ABI. The extension still
# links non-stable libtorch symbols, so it must always be rebuilt against the
# torch you run (see the build-isolation note in pyproject.toml).
py_limited_api = Version(torch.__version__) >= Version("2.6.0")

def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    if use_cuda:
        print("Compiling with cuda support")
    else:
        print("No cuda support")

    extra_link_args = ["-fopenmp"]
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            # Tune for the build host's ISA. torchbp is always compiled from
            # source against the local torch (see pyproject.toml), so the build
            # machine is the run machine, it is safe to use its full
            # instruction set. Helps with CPU ops.
            f"-march=native",
            "-fno-math-errno",
            "-fno-trapping-math",
            "-fdiagnostics-color=always",
            "-fopenmp",
            # Min Python version 3.9; only meaningful for an abi3 build.
            *(["-DPy_LIMITED_API=0x03090000"] if py_limited_api else []),
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
            "-DLIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS",
            "--use_fast_math",
            "-lineinfo"
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].extend(["-g", "-G"])
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    # CPU operator implementations live in csrc/cpu/, split per category
    # (mirroring the per-file layout of the CUDA ops in csrc/cuda/). They are
    # always compiled, so CPU ops are available in both CPU- and CUDA-enabled
    # builds.
    extensions_cpu_dir = os.path.join(extensions_dir, "cpu")
    sources += list(glob.glob(os.path.join(extensions_cpu_dir, "*.cpp")))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch", "numpy", "scipy"],
    extras_require = {
        # Test deps. expecttest is pulled in by torch.testing._internal
        # (TestCase / opcheck), which the test suite imports.
        'test': ["pytest", "expecttest"],
        'docs':  [
            "matplotlib >=3.5",
            "nbval >=0.9",
            "jupyter-client >=7.3.5",
            "sphinx-rtd-theme >=1.0",
            "sphinx >=4",
            "nbsphinx >= 0.8.9",
            "openpyxl >= 3",
            "lxml-html-clean >= 0.4.1",
            "numpydoc"]
    },
    description="Differentiable synthetic aperture radar library",
    long_description=open("Readme.md").read(),
    long_description_content_type="text/markdown",
    #url="",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
