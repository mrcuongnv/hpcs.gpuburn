# HPCS.GPUBurn
A HPCS-improved version of gpu_burn by Ville Timonen released at http://wili.cc/blog/gpu-burn.html

## Introduction

From http://wili.cc/blog/gpu-burn.html

> This program forks one process for each GPU on the machine, one process for keeping track of the GPU temperatures if available (e.g. Fermi Teslas don't have temp. sensors), and one process for reporting the progress. The GPU processes each allocate 90% of the free GPU memory, initialize 2 random 2048*2048 matrices, and continuously perform efficient CUBLAS matrix-matrix multiplication routines on them and store the results across the allocated memory. Both floats and doubles are supported. Correctness of the calculations is checked by comparing results of new calculations against a previous one -- on the GPU. This way the GPUs are 100% busy all the time and CPUs idle. The number of erroneous calculations is brought back to the CPU and reported to the user along with the number of operations performed so far and the GPU temperatures.

## Usage

```
Usage: hpcs.gpuburn [-d] [-t] [-s 10]

Options
    -d  Use double precision (default: single precision).
    -t  Print performance as Tflops (default: Gflops).
    -s  Benchmark duration in seconds (default: 10 seconds).
```

## Disclaimer

Use this tool at your own risks with extreme caution on system temprature. We DO NOT take any responsibility on the loss and damage of your devices or related things.
