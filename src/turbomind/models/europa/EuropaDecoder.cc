/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2022, SK Telecom Authored by A. Dialog
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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/models/multi_gpu_gpt/ParallelGptDecoder.cc

#include "src/turbomind/models/europa/EuropaDecoder.h"
#include "src/turbomind/kernels/layernorm_kernels.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/models/europa/europa_decoder_kernels.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/europa/europa_params.h"
#include "src/turbomind/models/llama/llama_utils.h"

namespace turbomind {

template<typename T>
EuropaDecoder<T>::EuropaDecoder(size_t                      head_num,
                              size_t                      kv_head_num,
                              size_t                      size_per_head,
                              size_t                      inter_size,
                              size_t                      num_layer,
                              const EuropaAttentionParams& attn_params,
                              float                       rmsnorm_eps,
                              NcclParam                   tensor_para,
                              cudaStream_t                stream,
                              cublasMMWrapper*            cublas_wrapper,
                              IAllocator*                 allocator,
                              bool                        is_free_buffer_after_forward,
                              int                         quant_policy):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    hidden_units_(head_num * size_per_head),
    rmsnorm_eps_(rmsnorm_eps),
    tensor_para_(tensor_para),
    data_type_(getTensorType<T>())
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    initialize(attn_params, kv_head_num, quant_policy);
}

template<typename T>
EuropaDecoder<T>::~EuropaDecoder()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    delete self_attention_layer_;
    delete gelu_ffn_layer_;
}

template<typename T>
void EuropaDecoder<T>::initialize(const EuropaAttentionParams& attn_params, size_t kv_head_num, int quant_policy)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    self_attention_layer_ = new EuropaDecoderSelfAttentionLayer<T>(head_num_,
                                                                  kv_head_num,
                                                                  size_per_head_,
                                                                  attn_params,
                                                                  tensor_para_,
                                                                  stream_,
                                                                  cublas_wrapper_,
                                                                  allocator_,
                                                                  is_free_buffer_after_forward_,
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
void EuropaDecoder<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void EuropaDecoder<T>::allocateBuffer(size_t batch_size)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    is_allocate_buffer_ = true;
}

template<typename T>
void EuropaDecoder<T>::freeBuffer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void EuropaDecoder<T>::forwardSelfAttn(const EuropaDecoder::Session&                   sess,
                                      T*                                             attn_io,
                                      const std::unordered_map<std::string, Tensor>* input_tensors,
                                      size_t                                         layer)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    TensorMap self_attention_input_tensors(*input_tensors);
    self_attention_input_tensors.insert("input_query",
                                        {MEMORY_GPU, data_type_, {sess.batch_size, hidden_units_}, attn_io});
    const int layer_id = layer;
    self_attention_input_tensors.insert("layer_id", {MEMORY_CPU, TYPE_INT32, {1}, &layer_id});

    auto& k_cache = *sess.k_cache;
    auto& v_cache = *sess.v_cache;

    TensorMap self_attention_output_tensors{
        {"attention_output", {MEMORY_GPU, data_type_, {sess.batch_size, hidden_units_}, attn_io}},
        {"key_cache", k_cache},
        {"value_cache", v_cache},
    };

    self_attention_layer_->forward(&self_attention_output_tensors,  //
                                   &self_attention_input_tensors,
                                   &sess.weights->at(layer)->self_attn_weights);
}

template<typename T>
void EuropaDecoder<T>::forwardFfn(const EuropaDecoder::Session& sess, T* ffn_io, size_t layer)
{
    TensorMap ffn_inputs{{"ffn_input", {MEMORY_GPU, data_type_, {sess.batch_size, hidden_units_}, ffn_io}}};
    TensorMap ffn_outputs{{"ffn_output", {MEMORY_GPU, data_type_, {sess.batch_size, hidden_units_}, ffn_io}}};
    gelu_ffn_layer_->forward(&ffn_outputs, &ffn_inputs, &sess.weights->at(layer)->ffn_weights);
}

template<typename T>
void EuropaDecoder<T>::forward(std::vector<Tensor>*                            output_tensors,
                              const std::vector<Tensor>*                      input_tensors,
                              const std::vector<EuropaDecoderLayerWeight<T>*>* decoder_layer_weights,
                            const T**                                            final_layernorm_weight,
                            const T**                                            final_layernorm_bias)
{
    FT_CHECK(false);
}

template<typename T>
void EuropaDecoder<T>::forward(std::unordered_map<std::string, Tensor>*        output_tensors,
                              const std::unordered_map<std::string, Tensor>*  input_tensors,
                              const std::vector<EuropaDecoderLayerWeight<T>*>* decoder_layer_weights,
                            const T**                                            final_layernorm_weight,
                            const T**                                            final_layernorm_bias)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    /**
     * input_tensors:
     *   \param decoder_input [batch_size, hidden_dims]
     *   \param sequence_lengths [batch_size] int
     *   \param output_norm_weight [hidden_dims]
     *   \param step [1] on cpu
     *   \param ite [1] on cpu
     *   \param finished [batch_size] bool
     *   \param total_padding_tokens [batch_size], int
     *   \param max_seq_len [1] on cpu
     *   \param masked_tokens [batch_size, memory_len] bool (optional), NOT USED YET
     *
     * output_tensors:
     *   \param decoder_output [batch_size, hidden_dimension]
     *   \param key_cache [batch_size] uint64_t
     *   \param value_cache [batch_size] uint64_t
     */

    // for the shape of key cache, refer to decoder_masked_multihead_attention_template.hpp

    Session sess{};
    sess.batch_size = input_tensors->at("decoder_input").shape[0];
    sess.weights    = decoder_layer_weights;
    allocateBuffer(sess.batch_size);

    sess.ite     = input_tensors->at("ite").getVal<const int>();
    sess.k_cache = &output_tensors->at("key_cache");
    sess.v_cache = &output_tensors->at("value_cache");

    sess.max_memory_len = input_tensors->at("max_seq_len").getVal<int>();

    T* decoder_input  = input_tensors->at("decoder_input").getPtr<T>();
    T* decoder_output = output_tensors->at("decoder_output").getPtr<T>();


    for (size_t layer = 0; layer < num_layer_; ++layer) {
        invokeGeneralLayerNorm(decoder_output,
                               decoder_input,
                               decoder_layer_weights->at(layer)->pre_self_attn_norm_weights,
                               decoder_layer_weights->at(layer)->pre_self_attn_norm_bias,
                               1e-05,
                               sess.batch_size,
                               hidden_units_,
                               nullptr,
                               nullptr,
                               0,
                               stream_);
        sync_check_cuda_error();

        forwardSelfAttn(sess, decoder_output, input_tensors, layer);
        invokeAddResidual(decoder_output, decoder_input, sess.batch_size, hidden_units_, stream_);
        sync_check_cuda_error();

        // {
        //     int num_element = 6144;
        //     half* data_host = new half[num_element];
        //     cudaD2Hcpy(data_host, (half *)(decoder_output), num_element);

        //     std::vector<float> vec_float(num_element);
        //     std::copy(data_host, data_host+num_element, vec_float.begin());

        //     std::string file_name = "/data/yocto_bak/analyse/decoder/lmdeploy_layer_0_attention_output.bin";
        //     std::ofstream outfile(file_name, std::ios::binary);
        //     if (outfile.is_open())
        //     {   
        //         std::cout << std::endl << "dumping to " << file_name << std::endl;
        //         outfile.write((char*)vec_float.data(), num_element * sizeof(float));
        //         outfile.close();
        //     }
        //     delete[] data_host;
        // }
        // exit(0);

        invokeGeneralLayerNorm(decoder_input,
                               decoder_output,
                               decoder_layer_weights->at(layer)->post_self_attn_norm_weights,
                               decoder_layer_weights->at(layer)->post_self_attn_norm_bias,
                               1e-05,
                               sess.batch_size,
                               hidden_units_,
                               nullptr,
                               nullptr,
                               0,
                               stream_);
        sync_check_cuda_error();

        ////////////////////////////////////////////
        /// feed-forward network
        TensorMap ffn_inputs{{"ffn_input", {MEMORY_GPU, data_type_, {sess.batch_size, hidden_units_}, decoder_input}}};
        TensorMap ffn_outputs{
            {"ffn_output", {MEMORY_GPU, data_type_, {sess.batch_size, hidden_units_}, decoder_input}}};
        gelu_ffn_layer_->forward(&ffn_outputs, &ffn_inputs, &decoder_layer_weights->at(layer)->ffn_weights);

        invokeAddBias(decoder_input, decoder_layer_weights->at(layer)->ffn_weights.dense_4h_to_h.bias, sess.batch_size, hidden_units_, stream_);
        sync_check_cuda_error();
        
        invokeAddResidual(decoder_input, decoder_output, sess.batch_size, hidden_units_, stream_);
        sync_check_cuda_error();

        // {
        //     int num_element = 6144;
        //     half* data_host = new half[num_element];
        //     cudaD2Hcpy(data_host, (half *)(decoder_input), num_element);

        //     std::vector<float> vec_float(num_element);
        //     std::copy(data_host, data_host+num_element, vec_float.begin());

        //     std::string file_name = "/data/yocto_bak/analyse/decoder/lmdeploy_layer_" + std::to_string(layer) + "_output.bin";
        //     std::ofstream outfile(file_name, std::ios::binary);
        //     if (outfile.is_open())
        //     {   
        //         std::cout << std::endl << "dumping to " << file_name << std::endl;
        //         outfile.write((char*)vec_float.data(), num_element * sizeof(float));
        //         outfile.close();
        //     }
        //     delete[] data_host;
        // }
    }
    
    invokeGeneralLayerNorm(decoder_output,
                           decoder_input,
                           *final_layernorm_weight,
                           *final_layernorm_bias,
                           1e-05,
                           sess.batch_size,
                           hidden_units_,
                           nullptr,
                           nullptr,
                           0,
                           stream_);
    sync_check_cuda_error();

    // {
    //     int num_element = 6144;
    //     half* data_host = new half[num_element];
    //     cudaD2Hcpy(data_host, (half *)(decoder_output), num_element);

    //     std::vector<float> vec_float(num_element);
    //     std::copy(data_host, data_host+num_element, vec_float.begin());

    //     std::string file_name = "/data/yocto_bak/analyse/dynamicDecode/final_layernorm_outputs_lmdeploy.bin";
    //     std::ofstream outfile(file_name, std::ios::binary);
    //     if (outfile.is_open())
    //     {   
    //         std::cout << std::endl << "dumping to " << file_name << std::endl;
    //         outfile.write((char*)vec_float.data(), num_element * sizeof(float));
    //         outfile.close();
    //     }
    //     delete[] data_host;
    // }
    // exit(0);
    
    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
}

template class EuropaDecoder<half>;
template class EuropaDecoder<float>;

}  // namespace turbomind
