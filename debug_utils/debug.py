import numpy as np
import os
from typing import Optional

def compute_mean_absolute_error(file_path1: str, file_path2: str, dtype, num_elements: int) -> Optional[float]:
    # Check if files exist
    if not os.path.isfile(file_path1) or not os.path.isfile(file_path2):
        print(f"One or both files do not exist: {file_path1}, {file_path2}")
        return None

    lmdeploy = np.fromfile(file_path1, dtype=dtype, count=num_elements)
    megatron = np.fromfile(file_path2, dtype=dtype, count=num_elements)

    # lmdeploy = lmdeploy.reshape([-1, 128])
    # megatron = megatron.reshape([-1, 128])

    import pdb;pdb.set_trace()

    return np.mean(np.abs(lmdeploy - megatron))

def construct_file_path(base_path: str, sub_dir: str, item: str) -> str:
    return os.path.join(base_path, sub_dir, f"{item}.bin")

def main():
    check_list_items = [
        # ("input_ids", np.int32, 79),
        # ("embedding_table", np.float32, 49152 * 6144),
        # ("position_embedding_table", np.float32, 8192 * 6144),
        # ("encoder_input", np.float32, 79 * 6144),
        # ("context_decoder_input", np.float32, 79 * 6144),
        # ("layer_0_pre_attention_layer_norm_weight", np.float32, 6144),
        # ("layer_0_pre_attention_layer_norm_bias", np.float32, 6144),
        # ("layer_0_pre_attention_layer_norm_output", np.float32, 78 * 6144),
        # ("layer_0_attention_output", np.float32, 78 * 6144),
        # ("layer_0_attention_qkv_output", np.float32, 78 * 8192),
        # ("layer_0_attention_qkv_weight", np.float32, 8192 * 6144),
        # ("layer_0_attention_qkv_bias", np.float32, 8192),
        # ("layer_0_attention_query_output", np.float32, 78 * 48 * 128),
        # ("layer_0_attention_key_output", np.float32, 78 * 8 * 128),
        # ("layer_0_attention_value_output", np.float32, 78 * 8 * 128),
        # ("layer_0_attention_linear_input", np.float32, 78 * 6144),
        # ("layer_0_attention_linear_weight", np.float32, 6144 * 6144),
        # ("layer_0_attention_linear_output", np.float32, 78 * 6144),
        # ("layer_0_post_attention_layernorm_input", np.float32, 78 * 6144),
        # ("layer_0_output", np.float32, 78 * 6144),
        # ("final_layernorm_input", np.float32, 78 * 6144),
        # ("final_layernorm_output", np.float32, 78 * 6144),
        # ("decoder_input", np.float32, 1 * 6144),
        # ("layer_0_layernorm_output", np.float32, 1 * 6144),
        # ("layer_0_pre_attention_layernorm_input", np.float32, 1 * 6144),
        # ("layer_0_pre_attention_layernorm_weight", np.float32, 6144 * 6144),
        # ("layer_0_pre_attention_layernorm_bias", np.float32, 1 * 6144),
        # ("layer_0_pre_attention_layernorm_output", np.float32, 1 * 6144),
        # ("layer_0_attention_qkv_input", np.float32, 6144),
        # ("layer_0_attention_qkv_weight", np.float32, 8192 * 6144),
        # ("layer_0_attention_qkv_bias", np.float32, 8192),
        # ("layer_0_attention_qkv_output", np.float32, 8192),
        # ("layer_0_attention_q_output", np.float32, 6144),
        # ("layer_0_attention_k_output", np.float32, 1024),
        # ("layer_0_attention_v_output", np.float32, 1024),
        ("layer_0_attention_output", np.float32, 6144),
    ]

    base_path = "/data/yocto_bak/analyse/"
    for item, dtype, elements_num in check_list_items:
        lmdeploy_path = construct_file_path(base_path, "lmdeploy", item)
        megatron_path = construct_file_path(base_path, "megatron", item)

        mean_absolute_error = compute_mean_absolute_error(lmdeploy_path, megatron_path, dtype, elements_num)

        if mean_absolute_error is not None:
            print(f"Mean absolute error for {item}: {mean_absolute_error}")

def test():
    base_path = "/data/yocto_bak/analyse/"

    lmdeploy_query_path = construct_file_path(base_path, "lmdeploy", "layer_0_attention_q_output")
    megatron_query_path = construct_file_path(base_path, "megatron", "layer_0_attention_q_output")

    lmdeploy_key_path = construct_file_path(base_path, "lmdeploy", "layer_0_attention_k_output")
    megatron_key_path = construct_file_path(base_path, "megatron", "layer_0_attention_k_output")

    lmdeploy_value_path = construct_file_path(base_path, "lmdeploy", "layer_0_attention_v_output")
    megatron_value_path = construct_file_path(base_path, "megatron", "layer_0_attention_v_output")

    lmdeploy_query_bias_path = construct_file_path(base_path, "lmdeploy", "layer_0_attention_q_bias_output")
    lmdeploy_key_bias_path = construct_file_path(base_path, "lmdeploy", "layer_0_attention_k_bias_output")
    lmdeploy_value_bias_path = construct_file_path(base_path, "lmdeploy", "layer_0_attention_v_bias_output")

    lmdeploy_output_path = construct_file_path(base_path, "lmdeploy", "layer_0_attention_output")
    megatron_output_path = construct_file_path(base_path, "megatron", "layer_0_attention_output")

    lmdeploy_query = np.fromfile(lmdeploy_query_path, dtype=np.float32, count=6144)
    megatron_query = np.fromfile(megatron_query_path, dtype=np.float32, count=6144)

    lmdeploy_key = np.fromfile(lmdeploy_key_path, dtype=np.float32, count=1024)
    megatron_key = np.fromfile(megatron_key_path, dtype=np.float32, count=1024)

    lmdeploy_value = np.fromfile(lmdeploy_value_path, dtype=np.float32, count=1024)
    megatron_value = np.fromfile(megatron_value_path, dtype=np.float32, count=1024)

    lmdeploy_query_bias = np.fromfile(lmdeploy_query_bias_path, dtype=np.float32, count=6144)
    lmdeploy_key_bias = np.fromfile(lmdeploy_key_bias_path, dtype=np.float32, count=1024)
    lmdeploy_value_bias = np.fromfile(lmdeploy_value_bias_path, dtype=np.float32, count=1024)

    lmdeploy_output = np.fromfile(lmdeploy_output_path, dtype=np.float32, count=6144)
    megatron_output = np.fromfile(megatron_output_path, dtype=np.float32, count=6144)

    lmdeploy_query = lmdeploy_query.reshape([6, 8, 128]).transpose([1,0,2])
    megatron_query = megatron_query.reshape([8, 6, 128])

    lmdeploy_key = lmdeploy_key.reshape([8, 128])
    megatron_key = megatron_key.reshape([8, 128])
    
    lmdeploy_value = lmdeploy_value.reshape([8, 128])
    megatron_value = megatron_value.reshape([8, 128])

    lmdeploy_query_bias = lmdeploy_query_bias.reshape([6, 8, 128]).transpose([1,0,2])
    lmdeploy_key_bias = lmdeploy_key_bias.reshape([8, 128])
    lmdeploy_value_bias = lmdeploy_value_bias.reshape([8, 128])

    lmdeploy_query = lmdeploy_query + lmdeploy_query_bias
    lmdeploy_key = lmdeploy_key + lmdeploy_key_bias
    lmdeploy_value = lmdeploy_value + lmdeploy_value_bias

    lmdeploy_output = lmdeploy_output.reshape([6144])
    megatron_output = megatron_output.reshape([6144])

    print(np.mean(np.abs(lmdeploy_query - megatron_query)))
    print(np.mean(np.abs(lmdeploy_key - megatron_key)))
    print(np.mean(np.abs(lmdeploy_value - megatron_value)))
    print(np.mean(np.abs(lmdeploy_output - megatron_output)))
    
    import pdb;pdb.set_trace()


if __name__ == "__main__":
    # main()
    test()
