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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/models/multi_gpu_gpt/ParallelGptDecoder.h

#include "src/turbomind/layers/BaseLayer.h"
#include "src/turbomind/models/star_coder/StarCoderDecoderLayerWeight.h"
#include "src/turbomind/models/star_coder/StarCoderDecoderSelfAttentionLayer.h"
#include "src/turbomind/models/star_coder/StarCoderFfnLayer.h"
#include "src/turbomind/models/star_coder/star_coder_params.h"
#include "src/turbomind/utils/custom_ar_comm.h"
#include "src/turbomind/utils/nccl_utils.h"

namespace turbomind {

template<typename T>
class StarCoderDecoder: public BaseLayer {
protected:
    void allocateBuffer() override;  // deprecated
    void allocateBuffer(size_t batch_size);
    void freeBuffer() override;
    void initialize(const StarCoderAttentionParams& attn_params, size_t kv_head_num, int quant_policy);

    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;
    size_t hidden_units_;
    float  rmsnorm_eps_;

    NcclParam tensor_para_;

    StarCoderDecoderSelfAttentionLayer<T>* self_attention_layer_{};
    StarCoderFfnLayer<T>*                  gelu_ffn_layer_{};

    const DataType data_type_;

    struct Session {
        size_t                                          batch_size;
        int                                             ite;
        size_t                                          max_memory_len;
        Tensor*                                         k_cache;
        Tensor*                                         v_cache;
        const std::vector<StarCoderDecoderLayerWeight<T>*>* weights;
    };

    void forwardSelfAttn(const Session&                                 sess,
                         T*                                             attn_io,
                         const std::unordered_map<std::string, Tensor>* input_tensors,
                         size_t                                         layer);

    void forwardFfn(const StarCoderDecoder::Session& sess, T* ffn_io, size_t layer);

public:
    StarCoderDecoder(size_t                      head_num,
                 size_t                      kv_head_num,
                 size_t                      size_per_head,
                 size_t                      inter_size,
                 size_t                      num_layer,
                 const StarCoderAttentionParams& attn_params,
                 float                       rmsnorm_eps,
                 NcclParam                   tensor_para,
                 cudaStream_t                stream,
                 cublasMMWrapper*            cublas_wrapper,
                 IAllocator*                 allocator,
                 bool                        is_free_buffer_after_forward,
                 int                         quant_policy);

    ~StarCoderDecoder() override;

    virtual void forward(std::unordered_map<std::string, Tensor>*        output_tensors,
                         const std::unordered_map<std::string, Tensor>*  input_tensors,
                         const std::vector<StarCoderDecoderLayerWeight<T>*>* decoder_layer_weights,
                        const T**                                            final_layernorm_weight,
                        const T**                                            final_layernorm_bias);

    virtual void forward(std::vector<Tensor>*                            output_tensors,
                         const std::vector<Tensor>*                      input_tensors,
                         const std::vector<StarCoderDecoderLayerWeight<T>*>* decoder_layer_weights,
                        const T**                                            final_layernorm_weight,
                        const T**                                            final_layernorm_bias);
};

}  // namespace turbomind
