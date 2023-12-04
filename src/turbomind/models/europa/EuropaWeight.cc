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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/models/multi_gpu_gpt/ParallelGptWeight.cc

#include "src/turbomind/models/europa/EuropaWeight.h"

namespace turbomind {

template<typename T>
EuropaWeight<T>::EuropaWeight(size_t     head_num,
                              size_t     kv_head_num,
                              size_t     size_per_head,
                              size_t     inter_size,
                              size_t     vocab_size,
                              size_t     num_layer,
                              bool       attn_bias,
                              WeightType weight_type,
                              int        group_size,
                              size_t     tensor_para_size,
                              size_t     tensor_para_rank):
    hidden_units_(head_num * size_per_head),
    inter_size_(inter_size),
    vocab_size_(vocab_size),
    vocab_size_padded_(vocab_size),
    num_layer_(num_layer),
    weight_type_(weight_type),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank)
{
    if (vocab_size_padded_ % tensor_para_size_ != 0) {
        vocab_size_padded_ = (vocab_size_padded_ + tensor_para_size_ - 1) / tensor_para_size_ * tensor_para_size_;
        TM_LOG_WARNING("pad vocab size from %d to %d", vocab_size_, vocab_size_padded_);
    }
    decoder_layer_weights.reserve(num_layer_);
    for (unsigned l = 0; l < num_layer_; ++l) {
        decoder_layer_weights.push_back(new EuropaDecoderLayerWeight<T>(head_num,
                                                                        kv_head_num,
                                                                        size_per_head,
                                                                        inter_size_,
                                                                        weight_type_,
                                                                        group_size,
                                                                        attn_bias,
                                                                        tensor_para_size_,
                                                                        tensor_para_rank_));
    }

    mallocWeights();
}

template<typename T>
EuropaWeight<T>::~EuropaWeight()
{
    cudaFree((void*)pre_decoder_embedding_table);
    cudaFree((void*)pre_decoder_position_embedding_table);
    cudaFree((void*)output_norm_weight);
    cudaFree((void*)output_norm_bias);
    cudaFree((void*)post_decoder_embedding_kernel);
    cudaFree((void*)post_decoder_position_embedding_kernel);

    pre_decoder_embedding_table            = nullptr;
    pre_decoder_position_embedding_table   = nullptr;
    output_norm_weight                     = nullptr;
    output_norm_bias                       = nullptr;
    post_decoder_embedding_kernel          = nullptr;
    post_decoder_position_embedding_kernel = nullptr;
}

template<typename T>
void EuropaWeight<T>::mallocWeights()
{
    // 尚未找到接入这个参数的方法，暂时用下面设置代替
    int max_output_len = 8192;

    deviceMalloc((T**)&pre_decoder_embedding_table, vocab_size_padded_ * hidden_units_);
    deviceMalloc((T**)&pre_decoder_position_embedding_table, max_output_len * hidden_units_);
    deviceMalloc((T**)&output_norm_weight, hidden_units_);
    deviceMalloc((T**)&output_norm_bias, hidden_units_);
    deviceMalloc((T**)&post_decoder_embedding_kernel, hidden_units_ * vocab_size_padded_);
    deviceMalloc((T**)&post_decoder_position_embedding_kernel, hidden_units_ * max_output_len);
}

template<typename T>
void EuropaWeight<T>::loadModel(std::string dir_path)
{
    FtCudaDataType model_file_type = FtCudaDataType::FP16;
    dir_path += '/';

    // 尚未找到接入这个参数的方法，暂时用下面设置代替
    int max_output_len = 8192;

    loadWeightFromBin((T*)pre_decoder_embedding_table,
                      {vocab_size_padded_ * hidden_units_},
                      dir_path + "word_embeddings.weight",
                      model_file_type);

    loadWeightFromBin((T*)pre_decoder_position_embedding_table,
                      {max_output_len * hidden_units_},
                      dir_path + "position_embeddings.weight",
                      model_file_type);

    loadWeightFromBin((T*)output_norm_weight, 
                      {hidden_units_}, 
                      dir_path + "final_layernorm.weight", 
                      model_file_type);    
                      
    loadWeightFromBin((T*)output_norm_bias, 
                      {hidden_units_}, 
                      dir_path + "final_layernorm.bias", 
                      model_file_type);

    loadWeightFromBin((T*)post_decoder_embedding_kernel,
                      {vocab_size_padded_ * hidden_units_},
                      dir_path + "word_embeddings.weight",
                      model_file_type);

    loadWeightFromBin((T*)post_decoder_position_embedding_kernel,
                      {max_output_len * hidden_units_},
                      dir_path + "position_embeddings.weight",
                      model_file_type);

    for (unsigned layer = 0; layer < num_layer_; ++layer) {
        decoder_layer_weights[layer]->loadModel(dir_path + "layers." + std::to_string(layer), model_file_type);
    }
}

template struct EuropaWeight<float>;
template struct EuropaWeight<half>;

}  // namespace turbomind
