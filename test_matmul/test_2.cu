#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    // 定义矩阵大小
    int m = 3, n = 4, k = 2;

    // 定义 alpha 和 beta
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 创建和初始化矩阵 A, B 和 C
    // float A[m * k] = {
    //                     1, 2,
    //                     4, 5,
    //                     7, 8
    //                  };
    // float B[k * n] = {
    //                     1, 1, 1, 1,
    //                     1, 1, 1, 1
    //                  };

    // 创建和初始化矩阵 A, B 和 C
    float A[k * n] = {
                        1, 1, 1, 1,
                        1, 1, 1, 1
                     };

    float B[m * k] = {
                        1, 2,
                        4, 5,
                        7, 8
                     };

    float C[m * n] = {0};

    float *d_A, *d_B, *d_C;

    // 初始化 CUDA 和 cuBLAS
    cudaMalloc((void **)&d_A, k * n * sizeof(float));
    cudaMalloc((void **)&d_B, m * k * sizeof(float));
    cudaMalloc((void **)&d_C, m * n * sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);

    // 将矩阵数据复制到设备上
    cudaMemcpy(d_A, A, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, m * k * sizeof(float), cudaMemcpyHostToDevice);

    // 执行矩阵乘法
    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 n, m, k,
                 &alpha,
                 d_A, CUDA_R_32F, n,
                 d_B, CUDA_R_32F, k,
                 &beta,
                 d_C, CUDA_R_32F, n,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

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
