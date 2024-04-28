!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git

!nvcc --version

pip install nvcc4jupyter

%load_ext nvcc4jupyter



%%cuda
#include <stdio.h>

#include <stdlib.h>

#include <time.h>

#include <cuda.h>

#define BLOCK_SIZE 16

__global__ void gpu_matrix_mult(int *a,int *b, int *c, int n)

{

int row = blockIdx.y * blockDim.y + threadIdx.y;

int col = blockIdx.x * blockDim.x + threadIdx.x;

int sum = 0;

if( col < n && row < n)

{

for(int i = 0; i < n; i++)

{

sum += a[row * n + i] * b[i * n + col];

}

c[row * n + col] = sum;

}

}

int main(int argc, char const *argv[])

{

int n;

for(n = 64; n <=8192 ; n *= 2) {

// allocate memory in host RAM, h_cc is used to store CPU result

int *h_a, *h_b, *h_c, *h_cc;

cudaMallocHost((void **) &h_a, sizeof(int)*n*n);

cudaMallocHost((void **) &h_b, sizeof(int)*n*n);

cudaMallocHost((void **) &h_c, sizeof(int)*n*n);

// initialize matrix A

for (int i = 0; i < n; ++i) {

for (int j = 0; j < n; ++j) {

h_a[i * n + j] = 2;

}

}

// initialize matrix B

for (int i = 0; i < n; ++i) {

for (int j = 0; j < n; ++j) {

h_b[i * n + j] = 3;

}

}

float naive_gpu_elapsed_time_ms;

// some events to count the execution time

//clock_t st, end;

cudaEvent_t start, stop;

cudaEventCreate(&start);

cudaEventCreate(&stop);

// Allocate memory space on the device

int *d_a, *d_b, *d_c;

cudaMalloc((void **) &d_a, sizeof(int)*n*n);

cudaMalloc((void **) &d_b, sizeof(int)*n*n);

cudaMalloc((void **) &d_c, sizeof(int)*n*n);

// copy matrix A and B from host to device memory

cudaMemcpy(d_a, h_a, sizeof(int)*n*n, cudaMemcpyHostToDevice);

cudaMemcpy(d_b, h_b, sizeof(int)*n*n, cudaMemcpyHostToDevice);

unsigned int grid_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

dim3 dimGrid(grid_cols, grid_rows);

dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

cudaEventRecord(start, 0);

gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);

cudaThreadSynchronize();

// time counting terminate

cudaEventRecord(stop, 0);

cudaEventSynchronize(stop);

// Transfer results from device to host

cudaMemcpy(h_cc, d_c, sizeof(int)*n*n, cudaMemcpyDeviceToHost);

// compute time elapsed on GPU computing

cudaEventElapsedTime(&naive_gpu_elapsed_time_ms, start, stop);

printf("Time elapsed on naive GPU matrix multiplication of %dx%d . %dx%d : %f ms.\n\n", n, n, n, n, naive_gpu_elapsed_time_ms);

// free memory

cudaFree(d_a);

cudaFree(d_b);

cudaFree(d_c);



}

return 0;

}
