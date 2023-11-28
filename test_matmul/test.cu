#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// 主函数
int main() {
    // 定义矩阵大小
    int m = 3, n = 4, k = 5;

    // 定义 alpha 和 beta
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 创建和初始化矩阵 A, B 和 C
    // float A[m * k] = {1, 2, 3, 4, 5,
    //                   6, 7, 8, 9, 1,
    //                   2, 3, 4, 5, 6};

    // float B[k * n] = {1, 1, 1, 1,
    //                   1, 1, 1, 1,
    //                   1, 1, 1, 1,
    //                   1, 1, 1, 1,
    //                   1, 1, 1, 1};

    float A[k * n] = {1, 0, 1, 1,
                      1, 1, 1, 1,
                      1, 1, 1, 1,
                      1, 1, 1, 1,
                      1, 1, 1, 1};

    float B[m * k] = {1, 2, 3, 4, 5,
                      6, 7, 8, 9, 1,
                      2, 3, 4, 5, 6};
    
    float C[m * n] = {0};
 
    float *d_A, *d_B, *d_C;

    // 初始化 CUDA 和 cuBLAS
    cudaMalloc((void **)&d_A, k * n * sizeof(float));
    cudaMalloc((void **)&d_B, m * k * sizeof(float));
    cudaMalloc((void **)&d_C, m * n * sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);

    // 将矩阵数据复制到设备上
    cudaMemcpy(d_A, A, n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * m * sizeof(float), cudaMemcpyHostToDevice);

    // 执行矩阵乘法
    cublasSgemmEx(handle, 
                CUBLAS_OP_N, 
                CUBLAS_OP_N, 
                n, // 4   output_row (origin output_col)
                m, // 3   output_col (origin output_row)
                k, // 5   
                &alpha, 
                d_A, // 5 * 4   origin 4 * 5
                n,   // 4 (origin leading dimension of A)
                d_B, // 3 * 5   origin 5 * 3
                k,   // 5 (origin leading dimension of B)
                &beta, 
                d_C, // 3 * 4   origin 4 * 3
                n);  // 4 (origin leading dimension of C)

    // 将结果复制回主机
    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result matrix C:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", C[i * n + j]);
        }
        printf("\n");
    }

    // 清理资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}
