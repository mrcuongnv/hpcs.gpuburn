# Honor explicit/common CUDA roots first, then derive the matching toolkit from
# nvcc. NVHPC places nvcc under compilers/bin and CUDA under cuda/<version>.
NVCC_ON_PATH := $(shell command -v nvcc 2>/dev/null)
NVCC_DISCOVERY := $(if $(filter undefined,$(origin NVCC)),$(NVCC_ON_PATH),$(NVCC))
NVCC_VERSION := $(if $(NVCC_DISCOVERY),$(shell $(NVCC_DISCOVERY) --version \
	2>/dev/null | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n 1))
NVCC_PREFIX := $(if $(NVCC_DISCOVERY),$(abspath $(dir $(NVCC_DISCOVERY))/..))
NVHPC_ROOT := $(if $(NVCC_DISCOVERY),$(abspath $(dir $(NVCC_DISCOVERY))/../..))
NVHPC_CUDA_PATH := $(if $(NVCC_VERSION),$(NVHPC_ROOT)/cuda/$(NVCC_VERSION))
CUDA_PATH_CANDIDATES := $(NVHPC_CUDA_HOME) $(CUDA_HOME) $(CUDA_PATH) \
	$(NVHPC_CUDA_PATH) $(NVCC_PREFIX) /usr/local/cuda

ifeq ($(origin CUDAPATH), undefined)
CUDAPATH := $(firstword $(foreach dir,$(CUDA_PATH_CANDIDATES),\
	$(if $(or $(wildcard $(dir)/include/cuda.h),\
			$(wildcard $(dir)/targets/*-linux/include/cuda.h)),$(dir))))
CUDAPATH := $(if $(CUDAPATH),$(CUDAPATH),/usr/local/cuda)
endif

NVCC ?= $(if $(wildcard $(CUDAPATH)/bin/nvcc),$(CUDAPATH)/bin/nvcc,$(or $(NVCC_ON_PATH),nvcc))
ifeq ($(origin CXX), default)
CXX := $(or $(shell command -v nvc++ 2>/dev/null),g++)
endif

# compute_60 PTX runs on Pascal and newer GPUs. CUDA 12.9 is the last toolkit
# release that can build this baseline and whose cuBLAS supports Pascal/Volta.
# With CUDA 13+, use COMPUTE=75 for Turing and newer GPUs.
COMPUTE ?= 60

# CUDA Toolkit packages may place headers/libraries under targets/<platform>.
# NVHPC additionally installs cuBLAS and cuBLASLt under math_libs/<version>.
CUDA_TARGET ?= $(firstword $(wildcard $(CUDAPATH)/targets/*-linux))
CUDA_MATH_PATH ?= $(subst /cuda/,/math_libs/,$(CUDAPATH))
CUDA_MATH_TARGET ?= $(firstword $(wildcard $(CUDA_MATH_PATH)/targets/*-linux))

CUDA_INCLUDE_CANDIDATES := \
	$(CUDAPATH)/include \
	$(if $(CUDA_TARGET),$(CUDA_TARGET)/include) \
	$(if $(CUDA_MATH_PATH),$(CUDA_MATH_PATH)/include) \
	$(if $(CUDA_MATH_TARGET),$(CUDA_MATH_TARGET)/include)

CUDA_LIBRARY_CANDIDATES := \
	$(CUDAPATH)/lib $(CUDAPATH)/lib64 \
	$(if $(CUDA_TARGET),$(CUDA_TARGET)/lib $(CUDA_TARGET)/lib64) \
	$(if $(CUDA_MATH_PATH),$(CUDA_MATH_PATH)/lib $(CUDA_MATH_PATH)/lib64) \
	$(if $(CUDA_MATH_TARGET),$(CUDA_MATH_TARGET)/lib $(CUDA_MATH_TARGET)/lib64)

CUDA_DRIVER_LIBRARY_CANDIDATES := \
	$(CUDAPATH)/lib/stubs $(CUDAPATH)/lib64/stubs \
	$(if $(CUDA_TARGET),$(CUDA_TARGET)/lib/stubs $(CUDA_TARGET)/lib64/stubs) \
	/usr/lib64 /usr/lib/x86_64-linux-gnu /usr/local/nvidia/lib /usr/local/nvidia/lib64

CUDA_INCLUDE_DIRS := $(sort $(foreach dir,$(CUDA_INCLUDE_CANDIDATES),\
	$(if $(wildcard $(dir)),$(dir))))
CUDA_LIBRARY_DIRS := $(sort $(foreach dir,$(CUDA_LIBRARY_CANDIDATES),\
	$(if $(wildcard $(dir)),$(dir))))
CUDA_DRIVER_LIBRARY_DIRS := $(sort $(foreach dir,$(CUDA_DRIVER_LIBRARY_CANDIDATES),\
	$(if $(wildcard $(dir)/libcuda.so),$(dir))))

CPPFLAGS += $(addprefix -I,$(CUDA_INCLUDE_DIRS)) -DMIN_COMPUTE=$(COMPUTE)
CXXFLAGS += -O3 -std=c++11 -Wall -Wextra
NVCCFLAGS += -arch=compute_$(COMPUTE) -ptx -Wno-deprecated-gpu-targets
LDFLAGS += $(addprefix -L,$(CUDA_DRIVER_LIBRARY_DIRS) $(CUDA_LIBRARY_DIRS))
# Never add the CUDA driver stub directory to rpath. At runtime, libcuda.so.1
# must resolve to the real NVIDIA driver installed on the compute node.
LDFLAGS += $(addprefix -Wl$(comma)-rpath$(comma),$(CUDA_LIBRARY_DIRS))
LDLIBS += -lcuda -lcublasLt -lcublas -lcudart

comma := ,

.PHONY: all drv clean print-config

all: drv

drv: gpu_burn

compare.ptx: compare.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

gpu_burn-drv.o: gpu_burn-drv.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

gpu_burn: compare.ptx gpu_burn-drv.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) gpu_burn-drv.o $(LDLIBS) -o $@

print-config:
	@echo "NVCC=$(NVCC)"
	@echo "NVCC_VERSION=$(NVCC_VERSION)"
	@echo "CUDAPATH=$(CUDAPATH)"
	@echo "CXX=$(CXX)"
	@echo "CUDA_TARGET=$(CUDA_TARGET)"
	@echo "CUDA_MATH_PATH=$(CUDA_MATH_PATH)"
	@echo "CUDA_MATH_TARGET=$(CUDA_MATH_TARGET)"
	@echo "CUDA_INCLUDE_DIRS=$(CUDA_INCLUDE_DIRS)"
	@echo "CUDA_LIBRARY_DIRS=$(CUDA_LIBRARY_DIRS)"
	@echo "CUDA_DRIVER_LIBRARY_DIRS=$(CUDA_DRIVER_LIBRARY_DIRS)"

clean:
	rm -f compare.ptx gpu_burn-drv.o gpu_burn
