/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/models/multi_gpu_gpt/ParallelGptContextDecoder.cc

#include "src/turbomind/models/europa/EuropaContextDecoder.h"
#include "src/turbomind/kernels/bert_preprocess_kernels.h"
#include "src/turbomind/kernels/layernorm_kernels.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/models/europa/EuropaContextDecoder.h"
#include "src/turbomind/models/europa/europa_decoder_kernels.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/debug_utils.h"

namespace turbomind {

template<typename T>
void EuropaContextDecoder<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void EuropaContextDecoder<T>::allocateBuffer(size_t batch_size, size_t num_token, size_t max_q_len, size_t max_kv_len)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    attention_mask_ = (T*)allocator_->reMalloc(attention_mask_, sizeof(T) * batch_size * max_q_len * max_kv_len, false);
    padding_offset_ = (int*)allocator_->reMalloc(padding_offset_, sizeof(int) * batch_size * max_q_len, false);
    cu_seqlens_     = (int*)allocator_->reMalloc(cu_seqlens_, sizeof(int) * (batch_size + 1), false);

    is_allocate_buffer_ = true;
}

template<typename T>
void EuropaContextDecoder<T>::freeBuffer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)&padding_offset_);
        allocator_->free((void**)&cu_seqlens_);
        allocator_->free((void**)&attention_mask_);
        allocator_->free((void**)&h_pinned_token_num_ptr_, true);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void EuropaContextDecoder<T>::initialize(const EuropaAttentionParams& attn_params,
                                            size_t                          kv_head_num,
                                            bool                            use_fmha,
                                            int                             quant_policy)
{
    h_pinned_token_num_ptr_ = (size_t*)allocator_->reMalloc(h_pinned_token_num_ptr_, sizeof(size_t), true, true);

    context_attention_layer_ = new EuropaContextAttentionLayer<T>(head_num_,
                                                                     kv_head_num,
                                                                     size_per_head_,
                                                                     attn_params,
                                                                     tensor_para_,
                                                                     stream_,
                                                                     cublas_wrapper_,
                                                                     allocator_,
                                                                     is_free_buffer_after_forward_,
                                                                     use_fmha,
                                                                     quant_policy);
    gelu_ffn_layer_ = new EuropaGeluFfnLayer<T>(head_num_,
                                                size_per_head_,
                                                inter_size_,
                                                stream_,
                                                cublas_wrapper_,
                                                allocator_,
                                                is_free_buffer_after_forward_);
}

template<typename T>
void EuropaContextDecoder<T>::forwardSelfAttn(const Session&                                 sess,
                                                 T*                                             attn_io,
                                                 const std::unordered_map<std::string, Tensor>* input_tensors,
                                                 int                                            layer,
                                                 bool                                           is_final)
{
    // TM_LOG_ERROR(__PRETTY_FUNCTION__);
    TensorMap self_attention_input_tensors{
        {"input_query", Tensor{MEMORY_GPU, data_type_, {sess.token_num, hidden_units_}, attn_io}},
        {"attention_mask",
         {MEMORY_GPU, data_type_, {sess.batch_size, 1, sess.max_query_len, sess.max_key_len}, attention_mask_}},
        {"layer_id", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &layer}},
        {"is_final_layer", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &is_final}},
        {"padding_offset", {MEMORY_GPU, TYPE_INT32, {sess.token_num}, padding_offset_}},
        {"cu_seqlens", {MEMORY_GPU, TYPE_INT32, {sess.batch_size + 1}, cu_seqlens_}},
        {"input_lengths", {MEMORY_GPU, TYPE_INT32, {sess.batch_size}, sess.input_length}},
        {"history_lengths", {MEMORY_GPU, TYPE_INT32, {sess.batch_size}, sess.history_length}},
        {"context_lengths", {MEMORY_GPU, TYPE_INT32, {sess.batch_size}, sess.context_length}},
        {"max_seq_len", input_tensors->at("max_seq_len")}};

    auto& k_cache = *sess.k_cache;
    auto& v_cache = *sess.v_cache;

    TensorMap self_attention_output_tensors{
        {"hidden_features", {MEMORY_GPU, data_type_, {sess.token_num, hidden_units_}, attn_io}},
        {"key_cache", k_cache},
        {"value_cache", v_cache},
    };

    context_attention_layer_->forward(&self_attention_output_tensors,  //
                                      &self_attention_input_tensors,
                                      &sess.weights->at(layer)->self_attn_weights);
}

template<typename T>
EuropaContextDecoder<T>::EuropaContextDecoder(size_t                          head_num,
                                                    size_t                          kv_head_num,
                                                    size_t                          size_per_head,
                                                    size_t                          inter_size,
                                                    size_t                          num_layer,
                                                    const EuropaAttentionParams& attn_params,
                                                    float                           rmsnorm_eps,
                                                    NcclParam                       tensor_para,
                                                    cudaStream_t                    stream,
                                                    cublasMMWrapper*                cublas_wrapper,
                                                    IAllocator*                     allocator,
                                                    bool                            is_free_buffer_after_forward,
                                                    bool                            use_fmha,
                                                    int                             quant_policy):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    hidden_units_(head_num * size_per_head),
    num_layer_(num_layer),
    rmsnorm_eps_(rmsnorm_eps),
    tensor_para_(tensor_para),
    data_type_(getTensorType<T>())
{
    initialize(attn_params, kv_head_num, use_fmha, quant_policy);
}

template<typename T>
EuropaContextDecoder<T>::~EuropaContextDecoder()
{
    delete context_attention_layer_;
    delete gelu_ffn_layer_;
    freeBuffer();
}

template<typename T>
void EuropaContextDecoder<T>::forward(std::vector<Tensor>*                                output_tensors,
                                         const std::vector<Tensor>*                          input_tensors,
                                         const std::vector<EuropaDecoderLayerWeight<T>*>* decoder_layer_weights,
                                         const T**                                            final_layernorm_weight,
                                         const T**                                            final_layernorm_bias)
{
    FT_CHECK(false);
}

template<typename T>
void EuropaContextDecoder<T>::forward(std::unordered_map<std::string, Tensor>*            output_tensors,
                                         const std::unordered_map<std::string, Tensor>*      input_tensors,
                                         const std::vector<EuropaDecoderLayerWeight<T>*>* decoder_layer_weights,
                                         const T**                                            final_layernorm_weight,
                                         const T**                                            final_layernorm_bias)
{
    /**
     * input tensors:
     *   \param decoder_input [num_token, hidden_units], float
     *   \param input_lengths [batch_size], int
     *   \param history_lengths [batch_size], int
     *   \param context_legnths [batch_size], int
     *   \param output_norm_weight [hidden_dims], float
     *   \param max_q_len [1], int on cpu
     *   \param max_kv_len [1], int on cpu
     *   \param max_seq_len [1], int on cpu
     *
     * output tensors:
     *   \param decoder_output [num_token, hidden_units],
     *   \param key_cache [num_layer, batch, local_head_num, size_per_head // x, max_seq_len, x]
     *   \param value_cache [num_layer, batch, local_head_num, max_seq_len, size_per_head]
     *   \param last_token_hidden_units [batch_size, hidden_units]
     */

    // printf("\n -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*");
    // printf("\n -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- EuropaContextDecoder<T>::forward *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* \n");  

    Session sess{};

    sess.token_num     = input_tensors->at("decoder_input").shape[0];
    sess.batch_size    = input_tensors->at("input_lengths").shape[0];
    sess.max_query_len = input_tensors->at("max_q_len").getVal<int>();
    sess.max_key_len   = input_tensors->at("max_kv_len").getVal<int>();
    sess.weights       = decoder_layer_weights;

    sess.input_length   = input_tensors->at("input_lengths").getPtr<int>();
    sess.history_length = input_tensors->at("history_lengths").getPtr<int>();
    sess.context_length = input_tensors->at("context_lengths").getPtr<int>();

    T* decoder_input_output = input_tensors->at("decoder_input").getPtr<T>();
    T* decoder_output       = output_tensors->at("decoder_output").getPtr<T>();

    sess.k_cache = &output_tensors->at("key_cache");
    sess.v_cache = &output_tensors->at("value_cache");

    // printf("\n -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- Paramer check *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* \n");
    // printf("sess.token_num : %d \n", int(sess.token_num));
    // printf("sess.batch_size : %d \n", int(sess.batch_size));
    // printf("sess.max_query_len : %d \n", int(sess.max_query_len));
    // printf("sess.max_key_len : %d \n", int(sess.max_key_len));
    // printf("sess.token_num : %d \n", int(sess.token_num));
    // printf(" -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* \n");

    allocateBuffer(sess.batch_size, sess.token_num, sess.max_query_len, sess.max_key_len);

    size_t tmp_token_num{};
    invokeGetPaddingOffsetAndCuSeqLens(h_pinned_token_num_ptr_,
                                       &tmp_token_num,  // updated token num
                                       padding_offset_,
                                       cu_seqlens_,
                                       input_tensors->at("input_lengths").getPtr<int>(),
                                       sess.batch_size,
                                       sess.max_query_len,
                                       stream_);
    sync_check_cuda_error();
    FT_CHECK(tmp_token_num == sess.token_num);

    invokeCreateCausalMasks(attention_mask_,
                            sess.input_length,
                            sess.context_length,
                            sess.max_query_len,
                            sess.max_key_len,
                            sess.batch_size,
                            stream_);
    sync_check_cuda_error();

    for (size_t layer = 0; layer < num_layer_; ++layer) {
        invokeGeneralLayerNorm(decoder_output,
                               decoder_input_output,
                               decoder_layer_weights->at(layer)->pre_self_attn_norm_weights,
                               decoder_layer_weights->at(layer)->pre_self_attn_norm_bias,
                               1e-05,
                               sess.token_num,
                               hidden_units_,
                               nullptr,
                               nullptr,
                               0,
                               stream_);
        sync_check_cuda_error();

        /////////////////////////////////////////////
        /// self-attention
        forwardSelfAttn(sess, decoder_output, input_tensors, layer, false);
        invokeAddResidual(decoder_output, decoder_input_output, sess.token_num, hidden_units_, stream_);
        sync_check_cuda_error();

        invokeGeneralLayerNorm(decoder_input_output,
                               decoder_output,
                               decoder_layer_weights->at(layer)->post_self_attn_norm_weights,
                               decoder_layer_weights->at(layer)->post_self_attn_norm_bias,
                               1e-05,
                               sess.token_num,
                               hidden_units_,
                               nullptr,
                               nullptr,
                               0,
                               stream_);
        sync_check_cuda_error();

        ////////////////////////////////////////////
        /// feed-forward network
        TensorMap ffn_inputs{{"ffn_input", {MEMORY_GPU, data_type_, {sess.token_num, hidden_units_}, decoder_input_output}}};
        TensorMap ffn_outputs{
            {"ffn_output", {MEMORY_GPU, data_type_, {sess.token_num, hidden_units_}, decoder_input_output}}};
        gelu_ffn_layer_->forward(&ffn_outputs, &ffn_inputs, &decoder_layer_weights->at(layer)->ffn_weights);

        invokeAddBias(decoder_input_output, decoder_layer_weights->at(layer)->ffn_weights.dense_4h_to_h.bias, sess.token_num, hidden_units_, stream_);
        sync_check_cuda_error();
        
        invokeAddResidual(decoder_input_output, decoder_output, sess.token_num, hidden_units_, stream_);
        sync_check_cuda_error();
    }
    
    invokeGeneralLayerNorm(decoder_output,
                           decoder_input_output,
                           *final_layernorm_weight,
                           *final_layernorm_bias,
                           1e-05,
                           sess.token_num,
                           hidden_units_,
                           nullptr,
                           nullptr,
                           0,
                           stream_);
    sync_check_cuda_error();

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
}

template class EuropaContextDecoder<float>;
template class EuropaContextDecoder<half>;

}  // namespace turbomind
