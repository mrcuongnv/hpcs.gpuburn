CUDAPATH ?= /usr/local/cuda
NVCC ?= $(CUDAPATH)/bin/nvcc
CXX ?= g++

# compute_60 PTX runs on Pascal and newer GPUs. CUDA 12.9 is the last toolkit
# release that can build this baseline and whose cuBLAS supports Pascal/Volta.
# With CUDA 13+, use COMPUTE=75 for Turing and newer GPUs.
COMPUTE ?= 60

CPPFLAGS += -I$(CUDAPATH)/include -DMIN_COMPUTE=$(COMPUTE)
CXXFLAGS += -O3 -std=c++11 -Wall -Wextra
NVCCFLAGS += -arch=compute_$(COMPUTE) -ptx
LDFLAGS += -L$(CUDAPATH)/lib64 -L$(CUDAPATH)/lib
LDFLAGS += -Wl,-rpath,$(CUDAPATH)/lib64 -Wl,-rpath,$(CUDAPATH)/lib
LDLIBS += -lcuda -lcublasLt -lcublas -lcudart

.PHONY: all drv clean

all: drv

drv: gpu_burn

compare.ptx: compare.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

gpu_burn-drv.o: gpu_burn-drv.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

gpu_burn: compare.ptx gpu_burn-drv.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) gpu_burn-drv.o $(LDLIBS) -o $@

clean:
	rm -f compare.ptx gpu_burn-drv.o gpu_burn
