%%writefile sequential.c

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void displayMatrix(int*, int, int);

int main() {
    clock_t t;
    t = clock();

    int m = 1000;
    int n = 1000;
    int k = 2000;

    int *a = (int *)malloc(n * k * sizeof(int));
    int *b = (int *)malloc(k * n * sizeof(int));
    int *c = (int *)malloc(n * m * sizeof(int));


    for (int i = 0; i < n * k; ++i) {
        a[i] = rand() % 10;
    }

    for (int i = 0; i < k * m; ++i) {
        b[i] = rand() % 10;
    }

    //printf("Matrix A:\n");
    //displayMatrix(a, n, k);

    //printf("Matrix B:\n");
    //displayMatrix(b, k, m);


    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            int sum = 0;
            for (int p = 0; p < k; ++p) {
                sum += a[i * k + p] * b[p * m + j];
            }
            c[i * m + j] = sum;
        }
    }

    //printf("Matrix C:\n");
    //displayMatrix(c, n, m);
    

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
