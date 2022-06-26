
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

//for shared memory
#define TileSize 12
#define BlockSize 12
//const for column size
#define N 12
//Const for row size
#define M 12


using namespace std;


cudaError_t transposeMatrix(int (&matrix_a)[N][M], int (&t_pose)[N][M]);
cudaError_t multiplication(int *matrix_a, int *matrix_b, int *matrix_c, size_t sz, dim3 dimsum, dim3 threads);
cudaError_t multiplication_sh(int *matrix_a, int *matrix_b, int *matrix_c, size_t sz, dim3 dimsum, dim3 threads);

__global__ void transposeKernal(int *matrix_a, int *transpose) {
	int i = threadIdx.x;
	int x = blockDim.x;
	int j = blockIdx.x;

	transpose[x * j + i] = matrix_a[x * i + j];
}

__global__ void multiplicationKernel(int *matrix_a, int *matrix_b, int *matrix_c) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	matrix_c[row * N + col] = 0;
	for(int i = 0; i < N; i++) {
		matrix_c[row * N + col] += matrix_a[threadIdx.x * blockDim.x + blockIdx.x] * matrix_b[threadIdx.y * blockDim.y + blockIdx.y];
	}

	__syncthreads();
	
	
}

__global__ void multiplication_sh_Kernel(int *matrix_a, int *matrix_b, int *matrix_c) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Statically allocated shared memory
	__shared__ int s_a[TileSize];
	__shared__ int s_b[TileSize];

	// Accumulate in temporary variable
	int tmp = 0;

	// Sweep tile across matrix
	for(int i = 0; i < N; i += blockDim.x) {
		// Load in elements for this tile
		s_a[threadIdx.y * blockDim.x + threadIdx.x] = matrix_a[row * N + i + threadIdx.x];
		s_b[threadIdx.y * blockDim.x + threadIdx.x] = matrix_b[i * N + threadIdx.y * N + col];

		// Wait for both tiles to be loaded in before doing computation
		__syncthreads();

		// Do matrix multiplication on the small matrix
		for(int j = 0; j < blockDim.x; j++) {
			tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];

		}

		// Wait for all threads to finish using current tiles before loading in new
		// ones
		__syncthreads();
	}

	// Write back results
	matrix_c[row * N + col] = tmp;
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
			matrix_b[i * M + j] = 2;// j + rand() / 100;
		}
	}

	cout << "Mat_B" << endl;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++) {
			cout << matrix_b[i * M + j] << " ";
		}
		cout << endl;
	}

	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++) {
			matrix_c[i * M + j] = 2;// j + rand() / 100;
		}
	}

	mainStatus = multiplication(matrix_b, matrix_c, matrix_d, size, dimGrid, dimBlock);
	if(mainStatus != cudaSuccess) {
		fprintf(stderr, "Its dead jim %s\n", cudaGetErrorString(mainStatus));
	}

	cout << "After MM" << endl;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++) {
			cout << matrix_d[i * M + j] << " ";
		}
		cout << endl;
	}

	free(matrix_b);
	free(matrix_c);
	free(matrix_d);

	//fill a matrix with random data
	//i = rowidx
	for(int i = 0; i < N; i++) {
		//j = col idx
		for(int j = 0; j < M; j++) {
			matrix_b[i * M + j] = j + rand() / 100;
		}
	}

	cout << "Matrix B" << endl;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++) {
			cout << matrix_b[i * M + j] << " ";
		}
		cout << endl;
	}

	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++) {
			matrix_c[i * M + j] = j + rand() / 100;
		}
	}

	cout << "Matrix C" << endl;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++) {
			cout << matrix_c[i * M + j] << " ";
		}
		cout << endl;
	}


	mainStatus = multiplication_sh(matrix_b, matrix_c, matrix_d, size, dimGrid, dimBlock);
	if(mainStatus != cudaSuccess) {
		fprintf(stderr, "Its dead jim %s\n", cudaGetErrorString(mainStatus));
	}

	cout << "After SharedMem MM" << endl;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++) {
			cout << matrix_d[i * M + j] << " ";
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
	
	multiplicationKernel << <N, M>> > (d_ma, d_mb, d_mc);

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

cudaError_t multiplication_sh(int *matrix_a, int *matrix_b, int *matrix_c, size_t sz, dim3 dimsum, dim3 threads) {

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

	multiplication_sh_Kernel << <dimsum, threads >> > (d_ma, d_mb, d_mc);

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


