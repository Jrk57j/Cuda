
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define N 10
#define M 10

using namespace std;

cudaError_t calculatePrimes(int *c, int *a, unsigned int size);

cudaError_t transposeMatrix(int (&matrix_a)[N][M], int (&t_pose)[N][M]);

__global__ void transposeKernal(int *matrix_a, int *transpose) {
	int i = threadIdx.x;
	int x = blockDim.x;
	int j = blockIdx.x;

	transpose[x * j + i] = matrix_a[x * i + j];
}

__global__ void primeKernel(int *c, int *a){
	int i = threadIdx.x;
	if (i % 2 != 0) {
		c[i] = (6 * a[i]) + 1;
		c[i + 1] = (6 * a[i] - 1);
	}
}

int main(){
	//const int arraySize = 50;
	/*int *num_holder = (int*)malloc(sizeof(int) * arraySize);
	int *prime_holder = (int*)malloc(sizeof(int) * arraySize);*/
	//prime_holder[arraySize + 1] += '\0';

	//insert numbers to a
	/*for (int i = 0; i < arraySize; i++) {
		num_holder[i] = i;
	}*/

	// Add vectors in parallel.
	//cudaError_t cudaStatus = calculatePrimes(prime_holder, num_holder, arraySize);
	/*if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "calculatePrimes failed!");
		return 1;
	}*/

	/*for (int i = 0; i < arraySize; i++) {
		printf("6(%d) + 1 = %d\n", num_holder[i], prime_holder[i]);
	}*/
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
   /* cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}*/

	size_t size = (N * M) * sizeof(int);
	
	cudaError_t mainStatus;
	
	int matrix_a[N][M];
	int matrix_b[N][M];
	int matrix_c[N][M];
	
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++) {
			matrix_a[i][j] = 1 + rand() / 100;
		}
	}

	mainStatus = transposeMatrix(matrix_a, matrix_b);
	if(mainStatus != cudaSuccess) {
		fprintf(stderr, "failed in main\n");
	}

	cout << "Before Transpose" << endl;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++) {
			cout << matrix_a[i][j] << " ";
		}
		cout << endl;
	}


	cout << "After Transpose" << endl;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++) {
			cout << matrix_b[i][j] << " ";
		}
		cout << endl;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t calculatePrimes(int *c, int *a, unsigned int size){
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	primeKernel<<<1, size>>>(dev_c, dev_a);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "primekernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching primekernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	
	return cudaStatus;
}


cudaError_t transposeMatrix(int (&matrix_a)[N][M], int(&t_pose)[N][M]) {
	
	cudaError_t status;

	int *dev_matrix;
	int *dev_transpose;
	size_t size = (N * M) * sizeof(int);

	status = cudaSetDevice(0);
	if(status != cudaSuccess) {
		fprintf(stderr,"Failed to set device\n");
	}

	status = cudaMalloc((void **) &dev_matrix, size);
	if(status != cudaSuccess) {
		fprintf(stderr, "Faield to allocate memory for dev_matrix\n");
	}

	status = cudaMemcpy(dev_matrix, matrix_a, size, cudaMemcpyHostToDevice);
	if(status != cudaSuccess) {
		fprintf(stderr, "Failed to copy matrix_a to dev_matrix and device\n");
		goto Error;
	}

	status = cudaMalloc((void **) &dev_transpose, size);
	if(status != cudaSuccess) {
		fprintf(stderr, "Faield to allocate memory for dev_matrix\n");
	}

	status = cudaMemcpy(dev_transpose, t_pose, size, cudaMemcpyHostToDevice);
	if(status != cudaSuccess) {
		fprintf(stderr, "Failed to copy t_pose to dev_transpose and device\n");
		goto Error;
	}

	transposeKernal << <N, M >> > (dev_matrix, dev_transpose);

	status = cudaGetLastError();
	if(status != cudaSuccess) {
		fprintf(stderr, "Failed to %s\n", cudaGetErrorString(status));
		goto Error;
	}

	status = cudaDeviceSynchronize();
	if(status != cudaSuccess) {
		fprintf(stderr, "Failed to sync device %s\n", cudaGetErrorString(status));
		goto Error;
	}

	status = cudaMemcpy(t_pose, dev_transpose, size, cudaMemcpyDeviceToHost);
	if(status != cudaSuccess) {
		fprintf(stderr, "Failed to copy dev_transpose to t_pose %s\n", cudaGetErrorString(status));
	}



Error:
	cudaFree(dev_matrix);
	cudaFree(dev_transpose);

	return status;
}
