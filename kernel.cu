
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

//for shared memory
#define TileSize 4

#define BlockSize 2
//const for column size
#define N 12
//Const for row size
#define M 8


using namespace std;


cudaError_t transposeMatrix(int (&matrix_a)[N][M], int (&t_pose)[N][M]);
cudaError_t multiplication(int *matrix_a, int *matrix_b, int *matrix_c, size_t sz, dim3 dimsum, dim3 blktheadsizething);

__global__ void transposeKernal(int *matrix_a, int *transpose) {
	int i = threadIdx.x;
	int x = blockDim.x;
	int j = blockIdx.x;

	transpose[x * j + i] = matrix_a[x * i + j];
}

__global__ void multiplicationKernel(int *matrix_a, int *matrix_b, int *matrix_c, dim3 dimsum, dim3 threads) {

	//TODO make shared mem
	int temp = 0;

	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++) {
			temp += matrix_a[threadIdx.x * blockDim.x + i] * matrix_b[threadIdx.x * blockDim.x + j];
			matrix_c[threadIdx.x * blockDim.x + i] = temp;
		}
	}
}


int main(){

	size_t size = (N * M) * sizeof(int);
	cudaError_t mainStatus;

	//naive apprach for matrix
	int matrix_a[N][M];
	int t_pose[N][M];

	dim3 dimGrid(N / TileSize, M / TileSize, 1);
	dim3 dimBlock(TileSize, BlockSize, 1);

	int *matrix_d = (int*) malloc(size);
	int *matrix_b = (int*) malloc(size);
	int *matrix_c = (int*) malloc(size);
	/*
	* Fill matrix for transpose
	*/
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++) {
			matrix_a[i][j] = 1 + rand() / 100;
		}
	}
	//fun transpose 
 	mainStatus = transposeMatrix(matrix_a, t_pose);
	if(mainStatus != cudaSuccess) {
		fprintf(stderr, "failed in main\n");
	}
	//print matrix before transpose for comparsion
	cout << "Before Transpose" << endl;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++) {
			cout << matrix_a[i][j] << " ";
		}
		cout << endl;
	}
	//print transposed matrix for comparions
	cout << "After Transpose" << endl;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++) {
			cout << t_pose[i][j] << " ";
		}
		cout << endl;
	}
	
	//fill a matrix with random data
	//i = rowidx
	for(int i = 0; i < N; i++) {
		//j = col idx
		for(int j = 0; j < M; j++) {
			matrix_b[i * M + j] = j + rand() / 100;
		}
	}

	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++) {
			matrix_c[i * M + j] = j + rand() / 100;
		}
	}

	mainStatus = multiplication(matrix_b, matrix_c, matrix_d, size, dimGrid, dimBlock);
	if(mainStatus != cudaSuccess) {
		fprintf(stderr, "Its dead jim %s\n", cudaGetErrorString(mainStatus));
	}

	cout << "After MM" << endl;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++) {
			cout << matrix_c[i * M + j] << " ";
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

cudaError_t multiplication(int *matrix_a, int *matrix_b, int *matrix_c, size_t sz, dim3 dimsum, dim3 threads) {

	int *d_ma;
	int *d_mb;
	int *d_mc;
	cudaError_t status;

	status = cudaSetDevice(0);
	if(status != cudaSuccess) {
		fprintf(stderr, "Could not obtain GPU %s\n", cudaGetErrorString(status));
	}


	status = cudaMalloc((int **) &d_ma, sz);
	if(status != cudaSuccess) {
		fprintf(stderr, "Could not allocate memory  %s\n", cudaGetErrorString(status));
		goto Error;
	}
	status = cudaMalloc((int **) &d_mb, sz);
	if(status != cudaSuccess) {
		fprintf(stderr, "Could not allocate memory  %s\n", cudaGetErrorString(status));
		goto Error;
	}
	status = cudaMalloc((int **) &d_mc, sz);
	if(status != cudaSuccess) {
		fprintf(stderr, "Could not allocate memory  %s\n", cudaGetErrorString(status));
		goto Error;
	}

	status = cudaMemcpy(d_ma, matrix_a, sz, cudaMemcpyHostToDevice);
	if(status != cudaSuccess) {
		fprintf(stderr, "Could not copy memory from host to device  %s\n", cudaGetErrorString(status));
	}

	status = cudaMemcpy(d_mb, matrix_b, sz, cudaMemcpyHostToDevice);
	if(status != cudaSuccess) {
		fprintf(stderr, "Could not copy memory from host to device  %s\n", cudaGetErrorString(status));
		goto Error;
	}

	status = cudaMemcpy(d_mc, matrix_c, sz, cudaMemcpyHostToDevice);
	if(status != cudaSuccess) {
		fprintf(stderr, "Could not copy memory from host to device  %s\n", cudaGetErrorString(status));
		goto Error;
	}
	
	multiplicationKernel << <dimsum, threads>> > (d_ma, d_mb, d_mc, dimsum, threads);

	status = cudaGetLastError();
	if(status != cudaSuccess) {
		fprintf(stderr, "Failed in device  %s\n", cudaGetErrorString(status));
		goto Error;
	}

	status = cudaDeviceSynchronize();
	if(status != cudaSuccess) {
		fprintf(stderr, "Failed to sync device threads went kaboom %s\n", cudaGetErrorString(status));
		goto Error;
	}

	status = cudaMemcpy(matrix_c, d_mc, sz, cudaMemcpyDeviceToHost);
	if(status != cudaSuccess) {
		fprintf(stderr, "Failed to copy dev to host  %s\n", cudaGetErrorString(status));
		goto Error;
	}

Error:
	cudaFree(d_ma);
	cudaFree(d_mb);
	cudaFree(d_mc);

	return status;
}


