CUDA_HOME=/usr/local/cuda
CXX=g++
NVCC=$(CUDA_HOME)/bin/nvcc
GCC5_HOME=/opt/gcc/5.4.0

CXXFLAGS=-O3 -Wno-unused-result -I${CUDA_HOME}/include
LDLIBS=-L${CUDA_HOME}/lib64/stubs -lcuda -L${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib -Wl,-rpath=${CUDA_HOME}/lib64 -Wl,-rpath=${CUDA_HOME}/lib -lcublas -lcudart

hpcs.gpuburn: gpuburn.o
	$(CXX) $(CXXFLAGS) -o hpcs.gpuburn gpuburn.o $(LDLIBS)

gpuburn.o: gpuburn.cpp compare.ptx
	$(CXX) $(CXXFLAGS) -c gpuburn.cpp

compare.ptx: compare.cu
	PATH="$(GCC5_HOME)/bin:$(PATH)" $(NVCC) -std=c++11 -ptx compare.cu -o compare.ptx

clean:
	rm -f compare.ptx *.o hpcs.gpuburn
