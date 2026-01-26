# cuQuantum Python JAX

cuQuantum Python JAX provides a JAX extension for cuQuantum Python. It exposes selected functionality of cuQuantum SDK in a JAX-compatible way that enables JAX frameworks to directly interface with the exposed cuQuantum API. In the current release, cuQuantum JAX exposes a JAX interface to the Operator Action API from the cuDensityMat library.

## Documentation

Please visit the [NVIDIA cuQuantum Python documentation](https://docs.nvidia.com/cuda/cuquantum/latest/python).

## Building and installing cuQuantum Python JAX

### Requirements

The install-time dependencies of the cuQuantum Python JAX package include:

* cuquantum-python-cu12~=26.1.0 for CUDA 12 or cuquantum-python-cu13~=26.1.0 for CUDA 13
* jax[cuda12-local]>=0.5,<0.7 for CUDA 12 or jax[cuda13-local]>=0.8,<0.9 for CUDA 13
* pybind11
* setuptools>=77.0.3

Note: 
1. cuQuantum Python JAX is only supported with CUDA 12 and CUDA 13.
2. cuQuantum Python JAX installation does not support build isolation. The user needs to pass in `--no-build-isolation` to `pip` when installing cuQuantum Python JAX.

#### Installation using `jax[cudaXX-local]`

`cuquantum-python-jax` depends explicitly on `jax[cudaXX-local]`. `pip install cuquantum-python-jax` will install `jax[cudaXX-local]`.

Using `jax[cudaXX-local]` assumes the user provides both cuDNN and the CUDA Toolkit. cuDNN is not a part of the CUDA Toolkit and requires an additional installation. The user must also specify `LD_LIBRARY_PATH`, including the library folders containing `libcudnn.so` and `libcupti.so`.

`libcupti.so` is provided by the CUDA Toolkit. If the CUDA Toolkit is installed under `/usr/local/cuda`, `libcupti.so` is located under `/usr/local/cuda/extras/CUPTI/lib64` and `LD_LIBRARY_PATH` should contain this path.

`libcudnn.so` is installed separately from the CUDA Toolkit. The default installation location is `/usr/local/cuda/lib64`, and `LD_LIBRARY_PATH` should contain this path.

Both `libcudnn.so` and `libcupti.so` are installable with pip:

```
pip install nvidia-cudnn-cu12
pip install nvidia-cuda-cupti-cu12
```

After installing cuDNN and cuPTI, the user may install `cuquantum-python-jax` with `pip` using either:

```
pip install --no-build-isolation cuquantum-python-jax
```

in which case the CUDA version will be detected, or one of

```
pip install --no-build-isolation cuquantum-python-cu12[jax]
pip install --no-build-isolation cuquantum-python-cu13[jax]
```

where the CUDA version is explicitly specified on cuquantum-python.

Note:
1. If cuDNN and cuPTI are installed with `pip`, the user does not need to specify library folders in `LD_LIBRARY_PATH`.
2. When the latter command `pip install --no-build-isolation cuquantum-python-cu12[jax]`/`pip install --no-build-isolation cuquantum-python-cu13[jax]` is used, `--no-build-isolation` applies to both cuquantum-python and cuquantum-python-jax. The user needs to ensure cuquantum-python's build dependencies are installed before the installation.

#### Installing from source

To install cuQuantum Python JAX from source, first compile cuQuantum Python from source using the [instructions on GitHub](https://github.com/NVIDIA/cuQuantum/blob/main/python/README.md). Once complete, navigate to `python/extensions`, then:

```
export CUDENSITYMAT_ROOT=...
pip install .
```

Where `CUDENSITYMAT_ROOT` is the path to the libraries parent directory. For example, if `CUDENSITYMAT_ROOT=/usr/local`, `libcudensitymat.so` would be found under `/usr/local/lib` or `/usr/local/lib64`.

## Running

### Requirements

Runtime dependencies of the cuQuantum Python package include:

* An NVIDIA GPU with compute capability 7.5+
* cuquantum-python-cu12~=26.1.0 for CUDA 12 or cuquantum-python-cu13~=26.1.0 for CUDA 13
* jax[cuda12-local]>=0.5,<0.7 for CUDA 12 or jax[cuda13-local]>=0.8,<0.9 for CUDA 13 
* pybind11

## Developer Notes

* cuQuantum Python JAX does not support editable installation.
* Both cuQuantum Python and cuQuantum Python JAX need to be installed into `site-packages` for proper import of the library.
* cuQuantum Python JAX assumes cuQuantum Python will be available under the current `site-packages` directory.
