%%cuda --name CUDA_TILING.cu

#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16

__global__ void matrixMul(int *a, int *b, int *c, int n, int k, int m) {
    __shared__ int tileA[TILE_SIZE][TILE_SIZE];
    __shared__ int tileB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    int sum = 0;
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < n && t * TILE_SIZE + tx < k) {
            tileA[ty][tx] = a[row * k + t * TILE_SIZE + tx];
        } else {
            tileA[ty][tx] = 0;
        }

        if (col < m && t * TILE_SIZE + ty < k) {
            tileB[ty][tx] = b[(t * TILE_SIZE + ty) * m + col];
        } else {
            tileB[ty][tx] = 0;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tileA[ty][i] * tileB[i][tx];
        }

        __syncthreads();
    }

    if (row < n && col < m) {
        c[row * m + col] = sum;
    }
}

void displayMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    
    clock_t t;
    t = clock();

    int n = 1000;  
    int m = 1000; 
    int k = 2000;
    
    int *a = (int *)malloc(n * k * sizeof(int));
    int *b = (int *)malloc(k * m * sizeof(int));
    int *c = (int *)malloc(n * m * sizeof(int));

    
    for (int i = 0; i < n * k; ++i) {
        a[i] = rand() % 10;
    }

    for (int i = 0; i < k * m; ++i) {
        b[i] = rand() % 10;
    }

    
    int *dev_a, *dev_b, *dev_c;
    cudaMalloc((void **)&dev_a, n * k * sizeof(int));
    cudaMalloc((void **)&dev_b, k * m * sizeof(int));
    cudaMalloc((void **)&dev_c, n * m * sizeof(int));

    
    cudaMemcpy(dev_a, a, n * k * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, k * m * sizeof(int), cudaMemcpyHostToDevice);

    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((m + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    
    matrixMul<<<grid, block>>>(dev_a, dev_b, dev_c, n, k, m);

    
    cudaMemcpy(c, dev_c, n * m * sizeof(int), cudaMemcpyDeviceToHost);

    
    //printf("Matrix A:\n");
    //displayMatrix(a, n, k);

    //printf("Matrix B:\n");
    //displayMatrix(b, k, m);

    //printf("Result Matrix C:\n");
    //displayMatrix(c, n, m);

    
    free(a);
    free(b);
    free(c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf("TIME TAKEN: %.2f\n", time_taken);

return 0;
}

