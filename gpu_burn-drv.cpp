/* 
 * Copyright (c) 2016, Ville Timonen
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the FreeBSD Project.
 */

#define SIZE 2048ul // Matrices are SIZE*SIZE..  2048^2 should be efficiently implemented in CUBLAS
#define DEFAULT_USEMEM 0.9 // Try to allocate 90% of available memory

// Standard GEMM operation count: SIZE^3 multiplies and SIZE^3 additions.
#define OPS_PER_MUL (2ull*SIZE*SIZE*SIZE)

#ifndef MIN_COMPUTE
#define MIN_COMPUTE 60
#endif

#include <cstdio>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <sys/types.h>
#include <signal.h>
#include <sys/wait.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include <cuda.h>
#include "cublasLt.h"
#include "cublas_v2.h"

enum BurnMode {
	MODE_FP32,
	MODE_FP64,
	MODE_FP16,
	MODE_BF16,
	MODE_FP8
};

const char *modeName(BurnMode mode) {
	switch (mode) {
	case MODE_FP32: return "FP32";
	case MODE_FP64: return "FP64";
	case MODE_FP16: return "FP16";
	case MODE_BF16: return "BF16";
	case MODE_FP8: return "FP8 E4M3";
	}
	return "UNKNOWN";
}

int modeMinCompute(BurnMode mode) {
	switch (mode) {
	case MODE_FP16: return 70;
	case MODE_BF16: return 80;
	case MODE_FP8: return 89;
	default: return MIN_COMPUTE;
	}
}

bool isLtMode(BurnMode mode) {
	return mode == MODE_FP16 || mode == MODE_BF16 || mode == MODE_FP8;
}

size_t modeInputElementSize(BurnMode mode) {
	switch (mode) {
	case MODE_FP64: return sizeof(double);
	case MODE_FP16:
	case MODE_BF16: return sizeof(uint16_t);
	case MODE_FP8: return sizeof(uint8_t);
	default: return sizeof(float);
	}
}

void checkError(CUresult rCode, const std::string &desc = "") {
	if (rCode == CUDA_SUCCESS)
		return;

	const char *name = "CUDA_ERROR_UNKNOWN";
	const char *message = "unknown CUDA driver error";
	cuGetErrorName(rCode, &name);
	cuGetErrorString(rCode, &message);
	throw ((desc.empty()) ?
			std::string("Error: ") :
			(std::string("Error in \"") + desc + "\": ")) +
		name + " (" + message + ")";
}

void checkError(cublasStatus_t rCode, const std::string &desc = "") {
	if (rCode == CUBLAS_STATUS_SUCCESS)
		return;

	throw ((desc.empty()) ?
			std::string("Error: ") :
			(std::string("Error in \"") + desc + "\": ")) +
		cublasGetStatusString(rCode);
}

class BurnTest {
public:
	virtual ~BurnTest() {}
	virtual uint64_t getErrors() = 0;
	virtual size_t getIters() = 0;
	virtual void initBuffers(const void *A, const void *B) = 0;
	virtual void compute() = 0;
	virtual void compare() = 0;
};

template <class T> class GPU_Test : public BurnTest {
public:
	GPU_Test(int dev, BurnMode mode, double useMemory) :
			d_mode(mode), d_devNumber(dev), d_iters(0), d_useMemory(useMemory),
			d_error(0), d_dev(0), d_ctx(0), d_module(0), d_function(0), d_Cdata(0),
			d_Adata(0), d_Bdata(0), d_faultyElemData(0), d_cublas(0) {
		checkError(cuDeviceGet(&d_dev, d_devNumber));
		checkError(cuCtxCreate(&d_ctx, 0, d_dev));
		bind();
		checkError(cublasCreate(&d_cublas), "init");
	}

	~GPU_Test() {
		if (!d_ctx)
			return;
		cuCtxSetCurrent(d_ctx);
		if (d_cublas)
			cublasDestroy(d_cublas);
		if (d_faultyElemData)
			cuMemFree(d_faultyElemData);
		if (d_Cdata)
			cuMemFree(d_Cdata);
		if (d_Adata)
			cuMemFree(d_Adata);
		if (d_Bdata)
			cuMemFree(d_Bdata);
		if (d_module)
			cuModuleUnload(d_module);
		cuCtxDestroy(d_ctx);
	}

	uint64_t getErrors() {
		uint64_t tempErrs = d_error;
		d_error = 0;
		return tempErrs;
	}

	size_t getIters() {
		return d_iters;
	}

	void bind() {
		checkError(cuCtxSetCurrent(d_ctx), "Bind CTX");
	}

	size_t totalMemory() {
		bind();
		size_t freeMem, totalMem;
		checkError(cuMemGetInfo(&freeMem, &totalMem));
		return totalMem;
	}

	size_t availMemory() {
		bind();
		size_t freeMem, totalMem;
		checkError(cuMemGetInfo(&freeMem, &totalMem));
		return freeMem;
	}

	void initBuffers(const void *A, const void *B) {
		bind();

		char deviceName[256];
		int major, minor;
		checkError(cuDeviceGetName(deviceName, sizeof(deviceName), d_dev), "device name");
		checkError(cuDeviceComputeCapability(&major, &minor, d_dev), "compute capability");
		const int minCompute = modeMinCompute(d_mode);
		if (major*10 + minor < minCompute)
			throw std::string("This build requires compute capability ") +
				std::to_string(minCompute/10) + "." + std::to_string(minCompute%10) +
				" or newer";

		const size_t available = availMemory();
		const size_t useBytes = (size_t)((double)available*d_useMemory);
		const size_t resultSize = sizeof(T)*SIZE*SIZE;
		if (useBytes < 4*resultSize)
			throw std::string("Not enough free GPU memory for input matrices and comparison results");
		d_iters = (useBytes - 2*resultSize)/resultSize; // Remove A and B sizes

		printf("Initialized device %d: %s (compute %d.%d), %zu MB total, "
				"%zu MB available, using %zu MB, %zu result matrices, %s\n",
				d_devNumber, deviceName, major, minor, totalMemory()/1024ul/1024ul,
				available/1024ul/1024ul, useBytes/1024ul/1024ul, d_iters,
				modeName(d_mode));
		checkError(cuMemAlloc(&d_Cdata, d_iters*resultSize), "C alloc");
		checkError(cuMemAlloc(&d_Adata, resultSize), "A alloc");
		checkError(cuMemAlloc(&d_Bdata, resultSize), "B alloc");
		checkError(cuMemAlloc(&d_faultyElemData, sizeof(uint64_t)), "faulty data");

		// Populating matrices A and B
		checkError(cuMemcpyHtoD(d_Adata, A, resultSize), "A -> device");
		checkError(cuMemcpyHtoD(d_Bdata, B, resultSize), "B -> device");
		initCompareKernel();
	}

	void compute() {
		bind();
		static const float alpha = 1.0f;
		static const float beta = 0.0f;
		static const double alphaD = 1.0;
		static const double betaD = 0.0;

		for (size_t i = 0; i < d_iters; ++i) {
			if (d_mode == MODE_FP64)
				checkError(cublasDgemm(d_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
							SIZE, SIZE, SIZE, &alphaD,
							(const double*)d_Adata, SIZE,
							(const double*)d_Bdata, SIZE,
							&betaD, 
							(double*)d_Cdata + i*SIZE*SIZE, SIZE), "DGEMM");
			else
				checkError(cublasSgemm(d_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
							SIZE, SIZE, SIZE, &alpha,
							(const float*)d_Adata, SIZE,
							(const float*)d_Bdata, SIZE,
							&beta, 
							(float*)d_Cdata + i*SIZE*SIZE, SIZE), "SGEMM");
		}
	}

	void initCompareKernel() {
		checkError(cuModuleLoad(&d_module, "compare.ptx"), "load module");
		checkError(cuModuleGetFunction(&d_function, d_module,
					d_mode == MODE_FP64 ? "compareD" : "compare"), "get func");
		checkError(cuFuncSetCacheConfig(d_function, CU_FUNC_CACHE_PREFER_L1), "L1 config");
	}

	void compare() {
		uint64_t faultyElems;
		void *kernelArgs[] = { &d_Cdata, &d_faultyElemData, &d_iters };
		checkError(cuMemsetD8(d_faultyElemData, 0, sizeof(faultyElems)), "memset");
		checkError(cuLaunchKernel(d_function,
					SIZE/g_blockSize, SIZE/g_blockSize, 1,
					g_blockSize, g_blockSize, 1,
					0, 0, kernelArgs, 0), "launch compare kernel");
		checkError(cuMemcpyDtoH(&faultyElems, d_faultyElemData, sizeof(faultyElems)), "read fault count");
		if (faultyElems)
			d_error += faultyElems;
	}

private:
	BurnMode d_mode;
	int d_devNumber;
	size_t d_iters;
	double d_useMemory;

	uint64_t d_error;

	static const int g_blockSize = 16;

	CUdevice d_dev;
	CUcontext d_ctx;
	CUmodule d_module;
	CUfunction d_function;

	CUdeviceptr d_Cdata;
	CUdeviceptr d_Adata;
	CUdeviceptr d_Bdata;
	CUdeviceptr d_faultyElemData;

	cublasHandle_t d_cublas;
};

void *devicePointer(CUdeviceptr pointer) {
	return reinterpret_cast<void*>(static_cast<uintptr_t>(pointer));
}

const void *constDevicePointer(CUdeviceptr pointer) {
	return reinterpret_cast<const void*>(static_cast<uintptr_t>(pointer));
}

class LtGPU_Test : public BurnTest {
public:
	LtGPU_Test(int dev, BurnMode mode, double useMemory) :
			d_mode(mode), d_devNumber(dev), d_iters(0), d_resultSize(sizeof(float)*SIZE*SIZE),
			d_inputSize(modeInputElementSize(mode)*SIZE*SIZE), d_useMemory(useMemory),
			d_error(0), d_dev(0), d_ctx(0), d_module(0), d_function(0), d_Cdata(0),
			d_Adata(0), d_Bdata(0), d_faultyElemData(0), d_scaleData(0),
			d_workspace(0), d_workspaceSize(32ul*1024ul*1024ul), d_lt(0),
			d_opDesc(0), d_aDesc(0), d_bDesc(0), d_cDesc(0) {
		checkError(cuDeviceGet(&d_dev, d_devNumber));
		checkError(cuCtxCreate(&d_ctx, 0, d_dev));
		bind();
		checkError(cublasLtCreate(&d_lt), "cuBLASLt init");
	}

	~LtGPU_Test() {
		if (!d_ctx)
			return;
		cuCtxSetCurrent(d_ctx);
		if (d_cDesc)
			cublasLtMatrixLayoutDestroy(d_cDesc);
		if (d_bDesc)
			cublasLtMatrixLayoutDestroy(d_bDesc);
		if (d_aDesc)
			cublasLtMatrixLayoutDestroy(d_aDesc);
		if (d_opDesc)
			cublasLtMatmulDescDestroy(d_opDesc);
		if (d_lt)
			cublasLtDestroy(d_lt);
		if (d_workspace)
			cuMemFree(d_workspace);
		if (d_scaleData)
			cuMemFree(d_scaleData);
		if (d_faultyElemData)
			cuMemFree(d_faultyElemData);
		if (d_Cdata)
			cuMemFree(d_Cdata);
		if (d_Adata)
			cuMemFree(d_Adata);
		if (d_Bdata)
			cuMemFree(d_Bdata);
		if (d_module)
			cuModuleUnload(d_module);
		cuCtxDestroy(d_ctx);
	}

	uint64_t getErrors() {
		uint64_t tempErrs = d_error;
		d_error = 0;
		return tempErrs;
	}

	size_t getIters() {
		return d_iters;
	}

	void initBuffers(const void *A, const void *B) {
		bind();

		char deviceName[256];
		int major, minor;
		checkError(cuDeviceGetName(deviceName, sizeof(deviceName), d_dev), "device name");
		checkError(cuDeviceComputeCapability(&major, &minor, d_dev), "compute capability");
		const int minCompute = modeMinCompute(d_mode);
		if (major*10 + minor < minCompute)
			throw std::string(modeName(d_mode)) + " Tensor Core mode requires compute capability " +
				std::to_string(minCompute/10) + "." + std::to_string(minCompute%10) + " or newer";

		initMatmul();

		size_t freeMemory, totalMemory;
		checkError(cuMemGetInfo(&freeMemory, &totalMemory));
		const size_t useBytes = (size_t)((double)freeMemory*d_useMemory);
		const size_t fixedBytes = 2*d_inputSize + d_workspaceSize;
		if (useBytes < fixedBytes + 2*d_resultSize)
			throw std::string("Not enough free GPU memory for inputs, workspace, and comparison results");
		d_iters = (useBytes - fixedBytes)/d_resultSize;

		printf("Initialized device %d: %s (compute %d.%d), %zu MB total, "
				"%zu MB available, using %zu MB, %zu result matrices, %s cuBLASLt\n",
				d_devNumber, deviceName, major, minor, totalMemory/1024ul/1024ul,
				freeMemory/1024ul/1024ul, useBytes/1024ul/1024ul, d_iters,
				modeName(d_mode));

		checkError(cuMemAlloc(&d_Cdata, d_iters*d_resultSize), "C alloc");
		checkError(cuMemAlloc(&d_Adata, d_inputSize), "A alloc");
		checkError(cuMemAlloc(&d_Bdata, d_inputSize), "B alloc");
		checkError(cuMemAlloc(&d_faultyElemData, sizeof(uint64_t)), "faulty data");
		checkError(cuMemAlloc(&d_workspace, d_workspaceSize), "cuBLASLt workspace");
		checkError(cuMemcpyHtoD(d_Adata, A, d_inputSize), "A -> device");
		checkError(cuMemcpyHtoD(d_Bdata, B, d_inputSize), "B -> device");

		if (d_mode == MODE_FP8) {
			const float scale = 1.0f;
			checkError(cuMemAlloc(&d_scaleData, sizeof(scale)), "FP8 scale alloc");
			checkError(cuMemcpyHtoD(d_scaleData, &scale, sizeof(scale)), "FP8 scale -> device");
			const void *scalePointer = constDevicePointer(d_scaleData);
			checkError(cublasLtMatmulDescSetAttribute(d_opDesc,
						CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scalePointer,
						sizeof(scalePointer)), "FP8 A scale");
			checkError(cublasLtMatmulDescSetAttribute(d_opDesc,
						CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scalePointer,
						sizeof(scalePointer)), "FP8 B scale");
		}

		initCompareKernel();
	}

	void compute() {
		bind();
		static const float alpha = 1.0f;
		static const float beta = 0.0f;
		for (size_t i = 0; i < d_iters; ++i) {
			void *result = devicePointer(d_Cdata + i*d_resultSize);
			checkError(cublasLtMatmul(d_lt, d_opDesc, &alpha,
						constDevicePointer(d_Adata), d_aDesc,
						constDevicePointer(d_Bdata), d_bDesc, &beta,
						result, d_cDesc, result, d_cDesc, &d_algo,
						devicePointer(d_workspace), d_workspaceSize, 0),
					"cuBLASLt matmul");
		}
	}

	void compare() {
		uint64_t faultyElems;
		void *kernelArgs[] = { &d_Cdata, &d_faultyElemData, &d_iters };
		checkError(cuMemsetD8(d_faultyElemData, 0, sizeof(faultyElems)), "memset");
		checkError(cuLaunchKernel(d_function,
					SIZE/g_blockSize, SIZE/g_blockSize, 1,
					g_blockSize, g_blockSize, 1,
					0, 0, kernelArgs, 0), "launch compare kernel");
		checkError(cuMemcpyDtoH(&faultyElems, d_faultyElemData, sizeof(faultyElems)), "read fault count");
		if (faultyElems)
			d_error += faultyElems;
	}

private:
	void bind() {
		checkError(cuCtxSetCurrent(d_ctx), "Bind CTX");
	}

	cudaDataType_t inputType() const {
		switch (d_mode) {
		case MODE_FP16: return CUDA_R_16F;
		case MODE_BF16: return CUDA_R_16BF;
		case MODE_FP8: return CUDA_R_8F_E4M3;
		default: return CUDA_R_32F;
		}
	}

	void initMatmul() {
		const cudaDataType_t type = inputType();
		checkError(cublasLtMatmulDescCreate(&d_opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
				"matmul descriptor");
		if (d_mode == MODE_FP8) {
			const cublasOperation_t transpose = CUBLAS_OP_T;
			checkError(cublasLtMatmulDescSetAttribute(d_opDesc,
						CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)),
					"FP8 transpose");
		}
		checkError(cublasLtMatrixLayoutCreate(&d_aDesc, type, SIZE, SIZE, SIZE),
				"A layout");
		checkError(cublasLtMatrixLayoutCreate(&d_bDesc, type, SIZE, SIZE, SIZE),
				"B layout");
		checkError(cublasLtMatrixLayoutCreate(&d_cDesc, CUDA_R_32F, SIZE, SIZE, SIZE),
				"C layout");

		cublasLtMatmulPreference_t preference = 0;
		checkError(cublasLtMatmulPreferenceCreate(&preference), "matmul preference");
		checkError(cublasLtMatmulPreferenceSetAttribute(preference,
					CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &d_workspaceSize,
					sizeof(d_workspaceSize)), "workspace preference");
		int returnedResults = 0;
		cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(d_lt, d_opDesc,
				d_aDesc, d_bDesc, d_cDesc, d_cDesc, preference, 1,
				&d_heuristic, &returnedResults);
		cublasLtMatmulPreferenceDestroy(preference);
		checkError(status, "matmul heuristic");
		if (!returnedResults)
			throw std::string("No cuBLASLt algorithm supports the requested mode on this GPU");
		d_algo = d_heuristic.algo;
	}

	void initCompareKernel() {
		checkError(cuModuleLoad(&d_module, "compare.ptx"), "load module");
		checkError(cuModuleGetFunction(&d_function, d_module, "compare"), "get func");
		checkError(cuFuncSetCacheConfig(d_function, CU_FUNC_CACHE_PREFER_L1), "L1 config");
	}

	BurnMode d_mode;
	int d_devNumber;
	size_t d_iters;
	size_t d_resultSize;
	size_t d_inputSize;
	double d_useMemory;
	uint64_t d_error;

	static const int g_blockSize = 16;

	CUdevice d_dev;
	CUcontext d_ctx;
	CUmodule d_module;
	CUfunction d_function;
	CUdeviceptr d_Cdata;
	CUdeviceptr d_Adata;
	CUdeviceptr d_Bdata;
	CUdeviceptr d_faultyElemData;
	CUdeviceptr d_scaleData;
	CUdeviceptr d_workspace;
	size_t d_workspaceSize;

	cublasLtHandle_t d_lt;
	cublasLtMatmulDesc_t d_opDesc;
	cublasLtMatrixLayout_t d_aDesc;
	cublasLtMatrixLayout_t d_bDesc;
	cublasLtMatrixLayout_t d_cDesc;
	cublasLtMatmulHeuristicResult_t d_heuristic;
	cublasLtMatmulAlgo_t d_algo;
};

// Returns the number of devices
int initCuda() {
	checkError(cuInit(0));
	int deviceCount = 0;
	checkError(cuDeviceGetCount(&deviceCount));

	if (!deviceCount)
		throw std::string("No CUDA devices");

	#ifdef USEDEV
	if (USEDEV >= deviceCount)
		throw std::string("Not enough devices for USEDEV");
	#endif

	return deviceCount;
}

struct ClientReport {
	uint64_t processed;
	uint64_t errors;
	int status;
};

bool writeAll(int fd, const void *buffer, size_t size) {
	const char *data = reinterpret_cast<const char*>(buffer);
	size_t remaining = size;
	while (remaining) {
		ssize_t written = write(fd, data, remaining);
		if (written < 0 && errno == EINTR)
			continue;
		if (written <= 0)
			return false;
		data += written;
		remaining -= written;
	}
	return true;
}

bool readAll(int fd, void *buffer, size_t size) {
	char *data = reinterpret_cast<char*>(buffer);
	size_t remaining = size;
	while (remaining) {
		ssize_t bytes = read(fd, data, remaining);
		if (bytes < 0 && errno == EINTR)
			continue;
		if (bytes <= 0)
			return false;
		data += bytes;
		remaining -= bytes;
	}
	return true;
}

bool writeReport(int fd, const ClientReport &report) {
	return writeAll(fd, &report, sizeof(report));
}

bool readReport(int fd, ClientReport *report) {
	return readAll(fd, report, sizeof(*report));
}

void startBurn(int index, int writeFd, const void *A, const void *B,
		BurnMode mode, double useMemory) {
	BurnTest *our;
	try {
		if (isLtMode(mode))
			our = new LtGPU_Test(index, mode, useMemory);
		else if (mode == MODE_FP64)
			our = new GPU_Test<double>(index, mode, useMemory);
		else
			our = new GPU_Test<float>(index, mode, useMemory);
		our->initBuffers(A, B);
	} catch (const std::string &e) {
		fprintf(stderr, "Couldn't init a GPU test: %s\n", e.c_str());
		ClientReport report = { 0, 0, -1 };
		writeReport(writeFd, report);
		exit(124);
	}

	// The actual work
	try {
		while (true) {
			our->compute();
			our->compare();
			ClientReport report = {
				static_cast<uint64_t>(our->getIters()),
				our->getErrors(),
				0
			};
			if (!writeReport(writeFd, report))
				exit(0);
		}
	} catch (const std::string &e) {
		fprintf(stderr, "Failure during compute: %s\n", e.c_str());
		ClientReport report = { 0, 0, -1 };
		writeReport(writeFd, report);
		exit(111);
	}
}

int pollTemp(pid_t *p) {
	int tempPipe[2];
	if (pipe(tempPipe) != 0) {
		*p = -1;
		return -1;
	}
	
	pid_t myPid = fork();

	if (!myPid) {
		close(tempPipe[0]);
		dup2(tempPipe[1], STDOUT_FILENO); // Stdout
		execlp("nvidia-smi", "nvidia-smi", "--query-gpu=temperature.gpu",
				"--format=csv,noheader,nounits", "-l", "5", NULL);
		fprintf(stderr, "Could not invoke nvidia-smi, no temps available\n");

		exit(0);
	}
	if (myPid < 0) {
		close(tempPipe[0]);
		close(tempPipe[1]);
		*p = -1;
		return -1;
	}

	*p = myPid;
	close(tempPipe[1]);

	return tempPipe[0];
}

bool updateTemps(int handle, std::vector<int> *temps) {
	const size_t readSize = 128;
	static size_t gpuIter = 0;
	char data[readSize];
	size_t curPos = 0;

	while (curPos + 1 < readSize) {
		ssize_t bytes = read(handle, data + curPos, 1);
		if (bytes < 0 && errno == EINTR)
			continue;
		if (bytes <= 0)
			return false;
		if (data[curPos++] == '\n')
			break;
	}
	data[curPos] = 0;

	char *end = NULL;
	long tempValue = strtol(data, &end, 10);
	if (end != data)
		temps->at(gpuIter) = static_cast<int>(tempValue);
	if (end != data || strstr(data, "N/A")) {
		gpuIter = (gpuIter+1)%(temps->size());
	}
	return true;
}

double monotonicSeconds() {
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return (double)now.tv_sec + (double)now.tv_nsec / 1000000000.0;
}

bool listenClients(const std::vector<int> &clientFd, const std::vector<pid_t> &clientPid, int runTime) {
	pid_t tempPid;
	int tempHandle = pollTemp(&tempPid);
	int maxHandle = tempHandle;

	for (size_t i = 0; i < clientFd.size(); ++i) {
		if (clientFd.at(i) > maxHandle)
			maxHandle = clientFd.at(i);
	}

	std::vector<int> clientTemp(clientFd.size(), 0);
	std::vector<uint64_t> clientErrors(clientFd.size(), 0);
	std::vector<uint64_t> clientCalcs(clientFd.size(), 0);
	std::vector<double> clientUpdateTime(clientFd.size(), 0.0);
	std::vector<double> clientGflops(clientFd.size(), 0.0);
	std::vector<bool> clientFaulty(clientFd.size(), false);
	std::vector<bool> clientAlive(clientFd.size(), true);
	std::vector<bool> clientDied(clientFd.size(), false);
	std::vector<bool> clientReported(clientFd.size(), false);

	const double startTime = monotonicSeconds();
	float nextReport = 10.0f;
	bool childReport = false;
	bool ranFullDuration = true;
	while (true) {
		double thisTime = monotonicSeconds();
		if (thisTime - startTime >= runTime)
			break;

		fd_set waitHandles;
		FD_ZERO(&waitHandles);
		if (tempHandle >= 0)
			FD_SET(tempHandle, &waitHandles);
		for (size_t i = 0; i < clientFd.size(); ++i)
			if (clientAlive.at(i))
				FD_SET(clientFd.at(i), &waitHandles);

		struct timeval timeout;
		timeout.tv_sec = 1;
		timeout.tv_usec = 0;
		int changeCount = select(maxHandle+1, &waitHandles, NULL, NULL, &timeout);
		if (changeCount < 0) {
			if (errno == EINTR)
				continue;
			perror("select");
			ranFullDuration = false;
			break;
		}

		thisTime = monotonicSeconds();
		for (size_t i = 0; i < clientFd.size(); ++i)
			if (clientAlive.at(i) && FD_ISSET(clientFd.at(i), &waitHandles)) {
				ClientReport report;
				if (!readReport(clientFd.at(i), &report)) {
					clientAlive.at(i) = false;
					clientDied.at(i) = true;
					continue;
				}

				if (report.status != 0) {
					clientAlive.at(i) = false;
					clientDied.at(i) = true;
					continue;
				}
				clientErrors.at(i) += report.errors;
				clientFaulty.at(i) = clientFaulty.at(i) || report.errors != 0;
				if (clientUpdateTime.at(i) != 0.0) {
					const double delta = thisTime - clientUpdateTime.at(i);
					clientGflops.at(i) = (double)report.processed * (double)OPS_PER_MUL /
						delta / 1000.0 / 1000.0 / 1000.0;
				}
				clientUpdateTime.at(i) = thisTime;
				clientCalcs.at(i) += report.processed;
				clientReported.at(i) = true;
				childReport = true;
			}

		if (tempHandle >= 0 && FD_ISSET(tempHandle, &waitHandles) &&
				!updateTemps(tempHandle, &clientTemp)) {
			close(tempHandle);
			tempHandle = -1;
		}

		if (childReport) {
			float elapsed = fminf((float)(thisTime-startTime)/(float)runTime*100.0f, 100.0f);
			printf("\r%.1f%%  ", elapsed);
			printf("proc'd: ");
			for (size_t i = 0; i < clientCalcs.size(); ++i) {
				printf("%llu (%.0f Gflop/s)%s ",
						(unsigned long long)clientCalcs.at(i), clientGflops.at(i),
						clientDied.at(i) ? " (DIED!)" : "");
				if (i != clientCalcs.size() - 1)
					printf("- ");
			}
			printf("  errors: ");
			for (size_t i = 0; i < clientErrors.size(); ++i) {
				printf("%llu%s ", (unsigned long long)clientErrors.at(i),
						clientErrors.at(i) ? " (WARNING!)" : "");
				if (i != clientCalcs.size() - 1)
					printf("- ");
			}
			printf("  temps: ");
			for (size_t i = 0; i < clientTemp.size(); ++i) {
				printf(clientTemp.at(i) != 0 ? "%d C " : "-- ", clientTemp.at(i));
				if (i != clientCalcs.size() - 1)
					printf("- ");
			}

			fflush(stdout);

			if (nextReport < elapsed) {
				nextReport = elapsed + 10.0f;
				time_t wallTime = time(NULL);
				char timeText[64];
				strftime(timeText, sizeof(timeText), "%Y-%m-%d %H:%M:%S", localtime(&wallTime));
				printf("\n\tSummary at: %s\n", timeText);
			}
		}

		bool oneAlive = false;
		for (size_t i = 0; i < clientAlive.size(); ++i)
			oneAlive = oneAlive || clientAlive.at(i);
		if (!oneAlive) {
			fprintf(stderr, "\n\nNo clients are alive; aborting early\n");
			ranFullDuration = false;
			break;
		}
	}

	printf("\nKilling processes.. ");
	fflush(stdout);
	for (size_t i = 0; i < clientPid.size(); ++i)
		kill(clientPid.at(i), 15);

	if (tempPid > 0)
		kill(tempPid, 15);
	if (tempHandle >= 0)
		close(tempHandle);

	while (wait(NULL) != -1);
	printf("done\n");

	printf("\nTested %d GPUs:\n", (int)clientPid.size());
	bool success = ranFullDuration;
	for (size_t i = 0; i < clientPid.size(); ++i) {
		const char *status = clientDied.at(i) ? "DIED" :
			(!clientReported.at(i) ? "NO RESULTS" : (clientFaulty.at(i) ? "FAULTY" : "OK"));
		printf("\tGPU %d: %s\n", (int)i, status);
		success = success && !clientDied.at(i) && clientReported.at(i) && !clientFaulty.at(i);
	}
	return success;
}

void initHostInputs(BurnMode mode, void *A, void *B) {
	srand(10);
	if (mode == MODE_FP64) {
		double *a = static_cast<double*>(A);
		double *b = static_cast<double*>(B);
		for (size_t i = 0; i < SIZE*SIZE; ++i) {
			a[i] = (double)(rand()%1000000)/100000.0;
			b[i] = (double)(rand()%1000000)/100000.0;
		}
	} else if (mode == MODE_FP32) {
		float *a = static_cast<float*>(A);
		float *b = static_cast<float*>(B);
		for (size_t i = 0; i < SIZE*SIZE; ++i) {
			a[i] = (float)((double)(rand()%1000000)/100000.0);
			b[i] = (float)((double)(rand()%1000000)/100000.0);
		}
	} else if (mode == MODE_FP16 || mode == MODE_BF16) {
		uint16_t *a = static_cast<uint16_t*>(A);
		uint16_t *b = static_cast<uint16_t*>(B);
		const uint16_t fp16Values[] = { 0x3800, 0x3c00, 0x3e00, 0x4000 };
		const uint16_t bf16Values[] = { 0x3f00, 0x3f80, 0x3fc0, 0x4000 };
		const uint16_t *values = mode == MODE_FP16 ? fp16Values : bf16Values;
		for (size_t i = 0; i < SIZE*SIZE; ++i) {
			a[i] = values[i%4];
			b[i] = values[(i/4)%4];
		}
	} else {
		uint8_t *a = static_cast<uint8_t*>(A);
		uint8_t *b = static_cast<uint8_t*>(B);
		const uint8_t fp8Values[] = { 0x30, 0x38, 0x3c, 0x40 };
		for (size_t i = 0; i < SIZE*SIZE; ++i) {
			a[i] = fp8Values[i%4];
			b[i] = fp8Values[(i/4)%4];
		}
	}
}

bool launch(int runLength, BurnMode mode, double useMemory) {
	system("nvidia-smi -L");

	// Initting A and B with random data
	const size_t inputSize = modeInputElementSize(mode)*SIZE*SIZE;
	void *A = malloc(inputSize);
	void *B = malloc(inputSize);
	if (!A || !B) {
		fprintf(stderr, "Could not allocate host input matrices\n");
		free(A);
		free(B);
		return false;
	}
	initHostInputs(mode, A, B);

	// Forking a process..  This one checks the number of devices to use,
	// returns the value, and continues to use the first one.
	int mainPipe[2];
	if (pipe(mainPipe) != 0) {
		perror("pipe");
		free(A);
		free(B);
		return false;
	}
	int readMain = mainPipe[0];
	std::vector<int> clientPipes;
	std::vector<pid_t> clientPids;
	clientPipes.push_back(readMain);

	pid_t myPid = fork();
	if (myPid < 0) {
		perror("fork");
		close(mainPipe[0]);
		close(mainPipe[1]);
		free(A);
		free(B);
		return false;
	}
	if (!myPid) {
		// Child
		close(mainPipe[0]);
		int writeFd = mainPipe[1];
		int devCount = -1;
		try {
			devCount = initCuda();
		} catch (const std::string &e) {
			fprintf(stderr, "Could not initialize CUDA: %s\n", e.c_str());
		}
		writeAll(writeFd, &devCount, sizeof(devCount));

		if (devCount > 0)
			startBurn(0, writeFd, A, B, mode, useMemory);

		close(writeFd);
		return false;
	} else {
		clientPids.push_back(myPid);
		bool spawnFailure = false;

		close(mainPipe[1]);
		int devCount = -1;
		if (!readAll(readMain, &devCount, sizeof(devCount)))
			devCount = -1;

		if (devCount <= 0) {
			fprintf(stderr, "No usable CUDA devices\n");
			waitpid(myPid, NULL, 0);
			close(readMain);
			free(A);
			free(B);
			return false;
		} else {
			for (int i = 1; i < devCount; ++i) {
				int slavePipe[2];
				if (pipe(slavePipe) != 0) {
					perror("pipe");
					spawnFailure = true;
					continue;
				}
				clientPipes.push_back(slavePipe[0]);

				pid_t slavePid = fork();
				if (slavePid < 0) {
					perror("fork");
					close(slavePipe[0]);
					close(slavePipe[1]);
					clientPipes.pop_back();
					spawnFailure = true;
					continue;
				}

				if (!slavePid) {
					// Child
					close(slavePipe[0]);
					try {
						initCuda();
						startBurn(i, slavePipe[1], A, B, mode, useMemory);
					} catch (const std::string &e) {
						fprintf(stderr, "Could not initialize GPU %d: %s\n", i, e.c_str());
						ClientReport report = { 0, 0, -1 };
						writeReport(slavePipe[1], report);
					}

					close(slavePipe[1]);
					return false;
				} else {
					clientPids.push_back(slavePid);
					close(slavePipe[1]);
				}
			}

			bool success = listenClients(clientPipes, clientPids, runLength);
			for (size_t i = 0; i < clientPipes.size(); ++i)
				close(clientPipes.at(i));
			free(A);
			free(B);
			return success && !spawnFailure;
		}
	}

	free(A);
	free(B);
	return false;
}

int main(int argc, char **argv) {
	int runLength = 10;
	BurnMode mode = MODE_FP32;
	bool durationSpecified = false;
	double useMemory = DEFAULT_USEMEM;

	for (int i = 1; i < argc; ++i) {
		const std::string arg(argv[i]);
		if (arg == "-d" || arg == "--double") {
			mode = MODE_FP64;
		} else if (arg == "-p" || arg == "--precision") {
			if (++i >= argc) {
				fprintf(stderr, "%s requires fp32, fp64, fp16, bf16, or fp8\n", arg.c_str());
				return 2;
			}
			const std::string precision(argv[i]);
			if (precision == "fp32")
				mode = MODE_FP32;
			else if (precision == "fp64")
				mode = MODE_FP64;
			else if (precision == "fp16")
				mode = MODE_FP16;
			else if (precision == "bf16")
				mode = MODE_BF16;
			else if (precision == "fp8")
				mode = MODE_FP8;
			else if (precision == "nvfp4") {
				fprintf(stderr, "NVFP4 requires a separate Blackwell block-scaled implementation\n");
				return 2;
			}
			else {
				fprintf(stderr, "Unknown precision: %s\n", precision.c_str());
				return 2;
			}
		} else if (arg == "-m" || arg == "--memory") {
			if (++i >= argc) {
				fprintf(stderr, "%s requires a percentage\n", arg.c_str());
				return 2;
			}
			char *end = NULL;
			double percent = strtod(argv[i], &end);
			if (!end || *end || percent <= 0.0 || percent >= 100.0) {
				fprintf(stderr, "Memory percentage must be greater than 0 and less than 100\n");
				return 2;
			}
			useMemory = percent / 100.0;
		} else if (arg == "-h" || arg == "--help") {
			printf("Usage: %s [-p|--precision fp32|fp64|fp16|bf16|fp8] "
					"[-m|--memory percent] [duration]\n", argv[0]);
			return 0;
		} else if (!durationSpecified) {
			char *end = NULL;
			long duration = strtol(argv[i], &end, 10);
			if (!end || *end || duration <= 0 || duration > INT_MAX) {
				fprintf(stderr, "Duration must be a positive integer number of seconds\n");
				return 2;
			}
			runLength = static_cast<int>(duration);
			durationSpecified = true;
		} else {
			fprintf(stderr, "Unexpected argument: %s\n", argv[i]);
			return 2;
		}
	}

	if (!durationSpecified)
		printf("Run length not specified in the command line.  Burning for 10 secs\n");

	return launch(runLength, mode, useMemory) ? 0 : 1;
}
