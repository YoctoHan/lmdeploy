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

#include "src/turbomind/models/star_coder/StarCoderFfnLayer.h"

namespace turbomind {

template<typename T>
void StarCoderFfnLayer<T>::forward(TensorMap* output_tensors,
                          const TensorMap* input_tensors,
                          const StarCoderFfnWeight<T>* ffn_weights)
{
    // input tensors:
    //      ffn_input [token_num, hidden_dimension],

    // output tensors:
    //      ffn_output [token_num, hidden_dimension],

    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() == 1);
    FT_CHECK(output_tensors->size() == 1);
    // FT_CHECK(isValidTokenNum(input_tensors->at(0).shape[0]));
    allocateBuffer(input_tensors->at("ffn_input").shape[0]);

    const int m = input_tensors->at("ffn_input").shape[0];
    T* output_tensor = (T*)output_tensors->at("ffn_output").data;
    const T* input_tensor = (const T*)input_tensors->at("ffn_input").data;

    // invokeMixWeightGemm(
    //     ffn_weights->intermediate_weight,
    //     weights_buf_,
    //     input_tensor,
    //     inter_buf_,
    //     inter_size_,
    //     m,
    //     hidden_units_,
    //     cublas_wrapper_,
    //     stream_
    // );

    // 需要确认配参
    // CUBLAS_WRAPPER->Gemm(CUBLAS_OP_N,
    //                      CUBLAS_OP_N,
    //                      N_SIZE,
    //                      M_SIZE,
    //                      K_SIZE,
    //                      ffn_weights.kernel,
    //                      N_SIZE,
    //                      input_tensor,
    //                      K_SIZE,
    //                      OUTPUT_TENSOR,
    //                      N_SIZE);

    // invokeAddBiasActivation(m, ffn_weights->intermediate_weight.bias);
    // sync_check_cuda_error();

    // invokeMixWeightGemm(
    //     ffn_weights->output_weight,
    //     weights_buf_,
    //     inter_buf_,
    //     output_tensor,
    //     hidden_units_,
    //     m,
    //     inter_size_,
    //     cublas_wrapper_,
    //     stream_
    // );

    // 需要确认配参
    // CUBLAS_WRAPPER->Gemm(CUBLAS_OP_N,
    //                      CUBLAS_OP_N,
    //                      N_SIZE,
    //                      M_SIZE,
    //                      K_SIZE,
    //                      ffn_weights.kernel,
    //                      N_SIZE,
    //                      INPUT_TENSOR,
    //                      K_SIZE,
    //                      OUTPUT_TENSOR,
    //                      N_SIZE);

    sync_check_cuda_error();
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template<typename T>
StarCoderFfnLayer<T>::StarCoderFfnLayer(size_t head_num,
                                        size_t size_per_head,
                                        size_t inter_size,
                                        cudaStream_t stream,
                                        cublasMMWrapper* cublas_wrapper,
                                        IAllocator* allocator,
                                        bool is_free_buffer_after_forward,
                                        bool sparse,
                                        int int8_mode):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    inter_size_(inter_size),
    int8_mode_(int8_mode)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
StarCoderFfnLayer<T>::StarCoderFfnLayer(StarCoderFfnLayer<T> const& ffn_layer):
    BaseLayer(ffn_layer.stream_,
              ffn_layer.cublas_wrapper_,
              ffn_layer.allocator_,
              ffn_layer.is_free_buffer_after_forward_,
              ffn_layer.cuda_device_prop_,
              ffn_layer.sparse_),
    max_token_num_(ffn_layer.max_token_num_),
    head_num_(ffn_layer.head_num_),
    size_per_head_(ffn_layer.size_per_head_),
    hidden_units_(ffn_layer.hidden_units_),
    inter_size_(ffn_layer.inter_size_),
    int8_mode_(ffn_layer.int8_mode_)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
StarCoderFfnLayer<T>::~StarCoderFfnLayer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void StarCoderFfnLayer<T>::allocateBuffer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_ == false) {
        inter_buf_ = (T*)allocator_->malloc(sizeof(T) * max_token_num_ * inter_size_, false);
        weights_buf_ = (T*)allocator_->malloc(sizeof(T) * inter_size_ * hidden_units_, false);
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void StarCoderFfnLayer<T>::allocateBuffer(size_t token_num)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    inter_buf_ = (T*)allocator_->reMalloc(inter_buf_, sizeof(T) * token_num * inter_size_, false);
    weights_buf_ = (T*)allocator_->reMalloc(weights_buf_, sizeof(T) * inter_size_ * hidden_units_, false);
    is_allocate_buffer_ = true;
}

template<typename T>
void StarCoderFfnLayer<T>::freeBuffer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&inter_buf_));
        allocator_->free((void**)(&weights_buf_));
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool StarCoderFfnLayer<T>::isValidTokenNum(size_t token_num)
{
    if (max_token_num_ < token_num) {
        max_token_num_ = token_num;
    }
    return true;
}

template class StarCoderFfnLayer<float>;
template class StarCoderFfnLayer<half>;
#ifdef ENABLE_BF16
template class StarCoderFfnLayer<__nv_bfloat16>;
#endif

template<typename T>
GeluFfnLayer<T>::GeluFfnLayer(size_t head_num,
                              size_t size_per_head,
                              size_t inter_size,
                              cudaStream_t stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator* allocator,
                              bool is_free_buffer_after_forward,
                              bool sparse,
                              int int8_mode):
    StarCoderFfnLayer<T>(head_num,
                         size_per_head,
                         inter_size,
                         stream,
                         cublas_wrapper,
                         allocator,
                         is_free_buffer_after_forward,
                         sparse,
                         int8_mode)
{
}

template<typename T>
GeluFfnLayer<T>::GeluFfnLayer(GeluFfnLayer<T> const& gelu_ffn_layer): StarCoderFfnLayer<T>(gelu_ffn_layer)
{
}

template<typename T>
void GeluFfnLayer<T>::invokeAddBiasActivation(const int m, const T* bias)
{
    // invokeAddBiasGeluV2<T>(inter_buf_, bias, nullptr, nullptr, m, inter_size_, stream_);
}

template class GeluFfnLayer<float>;
template class GeluFfnLayer<half>;
#ifdef ENABLE_BF16
template class GeluFfnLayer<__nv_bfloat16>;
#endif

}  // namespace turbomind
