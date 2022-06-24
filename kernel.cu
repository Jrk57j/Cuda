
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define N 10
#define M 10

using namespace std;


cudaError_t transposeMatrix(int (&matrix_a)[N][M], int (&t_pose)[N][M]);

__global__ void transposeKernal(int *matrix_a, int *transpose) {
	int i = threadIdx.x;
	int x = blockDim.x;
	int j = blockIdx.x;

	transpose[x * j + i] = matrix_a[x * i + j];
}


int main(){

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
