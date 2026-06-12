# HPCS.GPUBurn
A HPCS-improved version of gpu_burn by Ville Timonen released at http://wili.cc/blog/gpu-burn.html

## Introduction

From http://wili.cc/blog/gpu-burn.html

> This program forks one process for each GPU on the machine, one process for keeping track of the GPU temperatures if available (e.g. Fermi Teslas don't have temp. sensors), and one process for reporting the progress. The GPU processes each allocate 90% of the free GPU memory, initialize 2 random 2048*2048 matrices, and continuously perform efficient CUBLAS matrix-matrix multiplication routines on them and store the results across the allocated memory. Both floats and doubles are supported. Correctness of the calculations is checked by comparing results of new calculations against a previous one -- on the GPU. This way the GPUs are 100% busy all the time and CPUs idle. The number of erroneous calculations is brought back to the CPU and reported to the user along with the number of operations performed so far and the GPU temperatures.

## Usage

Syntax:

```
gpu_burn [-p|--precision fp32|fp64|fp16|bf16|fp8] [-m|--memory percent] [duration=10]
```

Options

* `-p precision`: select the GEMM input precision. Default is `fp32`.
* `-d`: compatibility alias for `--precision fp64`.
* `-m percent`: use this percentage of currently available GPU memory. Default is 90.
* `duration`: the running length in seconds. Default is 10 seconds.

The program returns a non-zero exit status if a GPU reports calculation errors,
a worker process dies, or CUDA initialization fails.

## SLURM

The included `gpu_burn.slurm` sample requests one GPU on one node in the
`compute` partition, builds the program, and runs a 10-minute FP32 burn:

```
sbatch gpu_burn.slurm
```

Override the runtime settings at submission:

```
sbatch --export=ALL,PRECISION=fp16,DURATION=600,MEMORY_PERCENT=90 gpu_burn.slurm
```

Change `#SBATCH --gpus-per-node=1` in the job file to request more GPUs. The
program automatically runs one worker for every GPU exposed to the job by
SLURM.

## Building and GPU support

Build with:

```
make
```

For systems using an NVIDIA HPC SDK environment module:

```
module load nvhpc
make
```

No build arguments are normally required in either case. The Makefile detects
the standard `/usr/local/cuda` installation, common CUDA environment variables,
and the CUDA version bundled with an NVHPC module. It also searches the NVHPC
`math_libs/<version>` tree and selects `nvc++` when it is provided by the
module.

When switching between CUDA installations, rebuild from clean sources:

```
make clean
make
```

Use `make print-config` to inspect the detected paths. Override `CUDAPATH`,
`NVCC`, or `CXX` only for unusual installations.

The CUDA Driver API library (`libcuda.so`) is supplied by the installed NVIDIA
driver rather than the CUDA Toolkit. When the development symlink is not
installed system-wide, the Makefile links against the toolkit's
`targets/<platform>/lib/stubs/libcuda.so`. The stub directory is deliberately
excluded from runtime `rpath`, so execution still uses the real driver library.
If `make print-config` reports an empty `CUDA_DRIVER_LIBRARY_DIRS`, locate the
stub and pass its directory explicitly:

```
find /opt/nvidia/hpc_sdk -path '*/lib/stubs/libcuda.so' -print
make CUDA_DRIVER_LIBRARY_DIRS=/path/containing/libcuda.so
```

The default build emits `compute_60` PTX. This supports Pascal, Volta, and newer
GPUs, and lets the installed NVIDIA driver JIT-compile the small comparison
kernel for newer architectures.

CUDA 12.9 is the newest toolkit line that supports both the Pascal/Volta
baseline and current GPUs. CUDA 13 removes offline compilation and library
support for pre-Turing GPUs. For a CUDA 13 build that only needs Turing and
newer GPUs, use:

```
make clean
make COMPUTE=75
```

| GPU generations | Recommended build |
| --- | --- |
| Pascal, Volta, Turing, Ampere, Ada, Hopper, Blackwell | CUDA 12.9, default `COMPUTE=60` |
| Turing and newer only | CUDA 13+, `COMPUTE=75` |

CUDA 12.9 cuBLAS is broadly forward-compatible through PTX JIT, but NVIDIA
does not guarantee performance on future GPU models. Prefer the CUDA 13+
profile for the newest hardware when Pascal/Volta support is not required.

## Precision modes

| Mode | GEMM path | Accumulation/output | Minimum GPU |
| --- | --- | --- | --- |
| `fp32` | cuBLAS SGEMM | FP32 | Pascal, compute 6.0 |
| `fp64` | cuBLAS DGEMM | FP64 | Pascal, compute 6.0 |
| `fp16` | cuBLASLt Tensor Core GEMM | FP32 | Volta, compute 7.0 |
| `bf16` | cuBLASLt Tensor Core GEMM | FP32 | Ampere, compute 8.0 |
| `fp8` | cuBLASLt E4M3 Tensor Core GEMM with explicit 1.0 input scales | FP32 | Ada/Hopper/Blackwell, compute 8.9 |

Examples:

```
./gpu_burn --precision fp16 60
./gpu_burn --precision bf16 --memory 80 60
./gpu_burn --precision fp8 60
```

The narrow-precision modes use aligned 2048x2048 matrices and a cuBLASLt
heuristic selected with a 32 MiB workspace. Repeated FP32 output matrices are
compared for correctness. The comparison kernel treats NaN and infinity as
errors.

NVFP4 is not exposed as a mode. Blackwell FP4 uses packed `CUDA_R_4F_E2M1`
values with block scaling rather than FP8-style tensorwide scalar scaling. It
should be implemented as a separate Blackwell-only workload when validating
the Blackwell FP4 data path is a requirement.

## Disclaimer

Use this tool at your own risk and monitor system temperature and power limits.
We do not take responsibility for loss or damage to devices or related systems.
