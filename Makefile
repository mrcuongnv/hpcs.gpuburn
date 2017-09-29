CUDAPATH=/usr/local/cuda

# Have this point to an old enough gcc (for nvcc)
GCCPATH=/opt/gcc/5.4.0

NVCC=nvcc
CCPATH=${GCCPATH}/bin

drv:
	PATH="${CUDAPATH}/bin:.:${CCPATH}:${PATH}" ${NVCC} -ptx compare.cu -o compare.ptx
	g++ -O3 -Wno-unused-result -I${CUDAPATH}/include -c gpu_burn-drv.cpp
	g++ -o gpu_burn gpu_burn-drv.o -O3 -L${CUDAPATH}/lib64/stubs -lcuda -L${CUDAPATH}/lib64 -L${CUDAPATH}/lib -Wl,-rpath=${CUDAPATH}/lib64 -Wl,-rpath=${CUDAPATH}/lib -lcublas -lcudart -o gpu_burn
