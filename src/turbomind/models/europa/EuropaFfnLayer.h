/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "src/turbomind/kernels/activation_kernels.h"
#include "src/turbomind/models/europa/EuropaLinear.h"
// #include "src/turbomind/kernels/matrix_vector_multiplication.h"
#include "src/turbomind/models/europa/EuropaDecoderLayerWeight.h"
#include "src/turbomind/layers/BaseLayer.h"
// #include "src/turbomind/layers/FfnWeight.h"
#include "src/turbomind/utils/memory_utils.h"
#include <vector>

namespace turbomind {

enum EuropaActivationType {
    EuropaGelu,
    EuropaRelu
};

template<typename T>
class EuropaFfnLayer: public BaseLayer {
private:
    // buffer handling
    size_t max_token_num_ = 0;

    // meta data
    size_t head_num_;
    size_t size_per_head_;

    // int8_mode_ == 1 for weight quantized only gemm for GPT
    int int8_mode_ = 0;

    // calculated data
    size_t hidden_units_;

    void allocateBuffer() override;
    void freeBuffer() override;
    bool isValidTokenNum(size_t token_num);
    void allocateBuffer(size_t token_num);
    
    EuropaLinear<T>   linear_;

protected:
    T* input_buf_ = nullptr;
    T* weights_buf_ = nullptr;
    size_t inter_size_;
    virtual void invokeEuropaAddBiasActivation(const int m, const T* bias) = 0;

public:
    EuropaFfnLayer(size_t head_num,
                      size_t size_per_head,
                      size_t inter_size,
                      cudaStream_t stream,
                      cublasMMWrapper* cublas_wrapper,
                      IAllocator* allocator,
                      bool is_free_buffer_after_forward,
                      bool sparse = false,
                      int int8_mode = 0);

    virtual ~EuropaFfnLayer();

    virtual void forward(TensorMap* output_tensors,
                         const TensorMap* input_tensors,
                         const EuropaFfnWeight<T>* ffn_weights);
};

template<typename T>
class EuropaGeluFfnLayer: public EuropaFfnLayer<T> {
public:
    EuropaGeluFfnLayer(size_t head_num,
                       size_t size_per_head,
                       size_t inter_size,   
                       cudaStream_t stream,
                       cublasMMWrapper* cublas_wrapper,
                       IAllocator* allocator,
                       bool is_free_buffer_after_forward,
                       bool sparse = false,
                       int int8_mode = 0);

    virtual ~EuropaGeluFfnLayer() = default;

protected:
    using EuropaFfnLayer<T>::stream_;

private:
    using EuropaFfnLayer<T>::input_buf_;
    void invokeEuropaAddBiasActivation(const int m, const T* bias) override;
};

template<typename T>
class EuropaFastGeluFfnLayer: public EuropaFfnLayer<T> {
public:
    EuropaFastGeluFfnLayer(size_t head_num,
                           size_t size_per_head,
                           size_t inter_size,
                           cudaStream_t stream,
                           cublasMMWrapper* cublas_wrapper,
                           IAllocator* allocator,
                           bool is_free_buffer_after_forward,
                           bool sparse = false,
                           int int8_mode = 0);

    virtual ~EuropaFastGeluFfnLayer() = default;

protected:
    using EuropaFfnLayer<T>::stream_;

private:
    using EuropaFfnLayer<T>::input_buf_;
    void invokeEuropaAddBiasActivation(const int m, const T* bias) override;
};

}  // namespace turbomind
