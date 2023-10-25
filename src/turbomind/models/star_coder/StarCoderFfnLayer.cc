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
    T* input_tensor = (T*)input_tensors->at("ffn_input").data;

    linear_.forward(input_buf_, input_tensor, m, ffn_weights->dense_h_to_4h);

    invokeAddBiasActivation(m, ffn_weights->dense_h_to_4h.bias);
    sync_check_cuda_error();
    
    linear_.forward(output_tensor, input_buf_, m, ffn_weights->dense_4h_to_h);

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
    int8_mode_(int8_mode),
    linear_(cublas_wrapper, stream)
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
        input_buf_ = (T*)allocator_->malloc(sizeof(T) * max_token_num_ * inter_size_, false);
        weights_buf_ = (T*)allocator_->malloc(sizeof(T) * inter_size_ * hidden_units_, false);
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void StarCoderFfnLayer<T>::allocateBuffer(size_t token_num)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    input_buf_ = (T*)allocator_->reMalloc(input_buf_, sizeof(T) * token_num * hidden_units_ * 4, false);
    weights_buf_ = (T*)allocator_->reMalloc(weights_buf_, sizeof(T) * hidden_units_ * hidden_units_ * 4, false);
    is_allocate_buffer_ = true;
}

template<typename T>
void StarCoderFfnLayer<T>::freeBuffer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&input_buf_));
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
void GeluFfnLayer<T>::invokeAddBiasActivation(const int m, const T* bias)
{
    invokeAddBiasGeluV2<T>(input_buf_, bias, nullptr, nullptr, nullptr, m, m, 6144 * 4, stream_);
}

template class GeluFfnLayer<float>;
template class GeluFfnLayer<half>;
#ifdef ENABLE_BF16
template class GeluFfnLayer<__nv_bfloat16>;
#endif

}  // namespace turbomind
