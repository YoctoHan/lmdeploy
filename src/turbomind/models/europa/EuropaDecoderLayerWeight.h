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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.h

#pragma once

#include "src/turbomind/models/llama/LlamaDenseWeight.h"

namespace turbomind {

template<typename T>
struct EuropaDecoderLayerWeight {
public:
    EuropaDecoderLayerWeight() = delete;
    EuropaDecoderLayerWeight(size_t     head_num,
                             size_t     kv_head_num,
                             size_t     size_per_head,
                             size_t     inter_size,
                             WeightType weight_type,
                             int        group_size,
                             bool       attn_bias,
                             size_t     tensor_para_size,
                             size_t     tensor_para_rank);
    ~EuropaDecoderLayerWeight();
    EuropaDecoderLayerWeight(const EuropaDecoderLayerWeight& other) = delete;
    EuropaDecoderLayerWeight& operator=(const EuropaDecoderLayerWeight& other) = delete;

    void loadModel(std::string dir_path, FtCudaDataType model_file_type);

    T*                          pre_self_attn_norm_weights{};
    T*                          pre_self_attn_norm_bias{};
    EuropaAttentionWeight<T>    self_attn_weights{};
    EuropaFfnWeight<T>          ffn_weights{};
    T*                          post_self_attn_norm_weights{};
    T*                          post_self_attn_norm_bias{};

private:
    size_t     head_num_;
    size_t     kv_head_num_;
    size_t     size_per_head_;
    size_t     hidden_units_;
    size_t     inter_size_;
    WeightType weight_type_;
    size_t     bit_size_;
    bool       attn_bias_;
    size_t     tensor_para_size_;
    size_t     tensor_para_rank_;
    bool       is_maintain_buffer_ = false;

    void mallocWeights();
};

}  // namespace turbomind
