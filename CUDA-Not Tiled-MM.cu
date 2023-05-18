%%cuda --name CUDA.cu

#include <stdio.h>
#include <stdlib.h>

void displayMatrix(int*, int, int);

__global__ void matrixMul(int *a, int *b, int *c, int n, int k, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < m) {
        int sum = 0;
        for (int p = 0; p < k; ++p) {
            sum += a[row * k + p] * b[p * m + col];
        }
        c[row * m + col] = sum;
    }
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
        a[i] = 1;
    }

    for (int i = 0; i < k * m; ++i) {
        b[i] = 1;
    }

    
    int *dev_a, *dev_b, *dev_c;
    cudaMalloc((void **)&dev_a, n * k * sizeof(int));
    cudaMalloc((void **)&dev_b, k * m * sizeof(int));
    cudaMalloc((void **)&dev_c, n * m * sizeof(int));

    
    cudaMemcpy(dev_a, a, n * k * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, k * m * sizeof(int), cudaMemcpyHostToDevice);

    
    dim3 block(16, 16);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    
    matrixMul<<<grid, block>>>(dev_a, dev_b, dev_c, n, k, m);

    
    cudaMemcpy(c, dev_c, n * m * sizeof(int), cudaMemcpyDeviceToHost);

    
    //printf("Matrix C:\n");
    //displayMatrix(c, n, m);

    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);

    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf("TIME TAKEN: %.2f\n", time_taken);

    return 0;
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

