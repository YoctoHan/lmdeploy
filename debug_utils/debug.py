import numpy as np
import os
from typing import Optional

def check(lmdeploy : np.array, megatron : np.array) -> float:
    return np.mean(np.abs(lmdeploy - megatron))

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
        # ("0108_layer_0_layer_input", np.float32, 78 * 6144),
        # ("0108_layer_0_layer_preatten_norm_output", np.float32, 78 * 6144),
        # ("0108_layer_0_query", np.float32, 78 * 6144),
        # ("0108_layer_0_key", np.float32, 78 * 1024),
        # ("0108_layer_0_value", np.float32, 78 * 1024),
        ("0108_layer_0_attention_linear_input", np.float32, 78 * 6144),
        ("0108_layer_0_attention_linear_weight", np.float32, 6144 * 6144),
        ("0108_layer_0_attention_linear_output", np.float32, 78 * 6144),
    ]

    base_path = "/data/yocto_bak/analyse/"
    for item, dtype, elements_num in check_list_items:
        lmdeploy_path = construct_file_path(base_path, "lmdeploy", item)
        megatron_path = construct_file_path(base_path, "megatron", item)

        mean_absolute_error = compute_mean_absolute_error(lmdeploy_path, megatron_path, dtype, elements_num)

        if mean_absolute_error is not None:
            print(f"Mean absolute error for {item}: {mean_absolute_error}")

def test_attention():
    base_path = "/data/yocto_bak/analyse/"

    lmdeploy_query_path = construct_file_path(base_path, "lmdeploy", "0108_layer_0_query")
    megatron_query_path = construct_file_path(base_path, "megatron", "0108_layer_0_query")

    lmdeploy_key_path = construct_file_path(base_path, "lmdeploy", "0108_layer_0_key")
    megatron_key_path = construct_file_path(base_path, "megatron", "0108_layer_0_key")

    lmdeploy_value_path = construct_file_path(base_path, "lmdeploy", "0108_layer_0_value")
    megatron_value_path = construct_file_path(base_path, "megatron", "0108_layer_0_value")

    # lmdeploy_query_bias_path = construct_file_path(base_path, "lmdeploy", "layer_0_attention_q_bias_output")
    # lmdeploy_key_bias_path = construct_file_path(base_path, "lmdeploy", "layer_0_attention_k_bias_output")
    # lmdeploy_value_bias_path = construct_file_path(base_path, "lmdeploy", "layer_0_attention_v_bias_output")

    # lmdeploy_output_path = construct_file_path(base_path, "lmdeploy", "layer_0_attention_output")
    # megatron_output_path = construct_file_path(base_path, "megatron", "layer_0_attention_output")

    lmdeploy_query = np.fromfile(lmdeploy_query_path, dtype=np.float32, count=78 * 6 * 8 * 128)
    megatron_query = np.fromfile(megatron_query_path, dtype=np.float32, count=78 * 6 * 8 * 128)

    lmdeploy_key = np.fromfile(lmdeploy_key_path, dtype=np.float32, count=78 * 1 * 8 * 128)
    megatron_key = np.fromfile(megatron_key_path, dtype=np.float32, count=78 * 1 * 8 * 128)

    lmdeploy_value = np.fromfile(lmdeploy_value_path, dtype=np.float32, count=78 * 1 * 8 * 128)
    megatron_value = np.fromfile(megatron_value_path, dtype=np.float32, count=78 * 1 * 8 * 128)

    # lmdeploy_query_bias = np.fromfile(lmdeploy_query_bias_path, dtype=np.float32, count=6144)
    # lmdeploy_key_bias = np.fromfile(lmdeploy_key_bias_path, dtype=np.float32, count=1024)
    # lmdeploy_value_bias = np.fromfile(lmdeploy_value_bias_path, dtype=np.float32, count=1024)

    # lmdeploy_output = np.fromfile(lmdeploy_output_path, dtype=np.float32, count=6144)
    # megatron_output = np.fromfile(megatron_output_path, dtype=np.float32, count=6144)

    lmdeploy_query = lmdeploy_query.reshape([8, 6, 78, 128]).transpose([2, 0, 1, 3])
    megatron_query = megatron_query.reshape([78, 8, 6, 128])

    lmdeploy_key = lmdeploy_key.reshape([8, 78, 128]).transpose([1, 0, 2])
    megatron_key = megatron_key.reshape([78, 8, 128])
    
    lmdeploy_value = lmdeploy_value.reshape([8, 78, 128]).transpose([1, 0, 2])
    megatron_value = megatron_value.reshape([78, 8, 128])

    # lmdeploy_query_bias = lmdeploy_query_bias.reshape([6, 8, 128]).transpose([1,0,2])
    # lmdeploy_key_bias = lmdeploy_key_bias.reshape([8, 128])
    # lmdeploy_value_bias = lmdeploy_value_bias.reshape([8, 128])

    # lmdeploy_query = lmdeploy_query + lmdeploy_query_bias
    # lmdeploy_key = lmdeploy_key + lmdeploy_key_bias
    # lmdeploy_value = lmdeploy_value + lmdeploy_value_bias

    # lmdeploy_output = lmdeploy_output.reshape([6144])
    # megatron_output = megatron_output.reshape([6144])

    print(np.mean(np.abs(lmdeploy_query - megatron_query)))
    print(np.mean(np.abs(lmdeploy_key - megatron_key)))
    print(np.mean(np.abs(lmdeploy_value - megatron_value)))
    # print(np.mean(np.abs(lmdeploy_output - megatron_output)))

    # qk = np.sum([megatron_query[0][0][i] * megatron_key[0][i] for i in range(128)])
    
    import pdb;pdb.set_trace()

def test_cache():
    base_path = "/data/yocto_bak/analyse/"

    lmdeploy_kcache_path = construct_file_path(base_path, "lmdeploy", "keyCache0108")
    megatron_kcache_path = construct_file_path(base_path, "megatron", "k_cache")

    lmdeploy_vcache_path = construct_file_path(base_path, "lmdeploy", "v_cache")
    megatron_vcache_path = construct_file_path(base_path, "megatron", "v_cache")

    lmdeploy_kcache = np.fromfile(lmdeploy_kcache_path, dtype=np.float16, count=40 * 8192 * 8 * 128)
    megatron_kcache = np.fromfile(megatron_kcache_path, dtype=np.float32, count=40 * 78 * 8 * 128)

    lmdeploy_vcache = np.fromfile(lmdeploy_vcache_path, dtype=np.float32, count=40 * 78 * 8 * 128)
    megatron_vcache = np.fromfile(megatron_vcache_path, dtype=np.float32, count=40 * 78 * 8 * 128)
    
    lmdeploy_kcache = lmdeploy_kcache.reshape([40, 8192, 8, 128])
    megatron_kcache = megatron_kcache.reshape([40, 78, 8, 128])
    
    lmdeploy_vcache = lmdeploy_vcache.reshape([40, 78, 8, 128])
    megatron_vcache = megatron_vcache.reshape([40, 78, 8, 128])
    
    import pdb;pdb.set_trace()

def test():
    base_path = "/data/yocto_bak/analyse/"

    lmdeploy_key_path = construct_file_path(base_path, "lmdeploy", "layer_1_key")
    megatron_key_path = construct_file_path(base_path, "megatron", "layer_1_key")

    lmdeploy_value_path = construct_file_path(base_path, "lmdeploy", "value_check")
    megatron_value_path = construct_file_path(base_path, "megatron", "value_check")

    lmdeploy_key = np.fromfile(lmdeploy_key_path, dtype=np.float32, count=78 * 8 * 128)
    megatron_key = np.fromfile(megatron_key_path, dtype=np.float32, count=78 * 8 * 128)

    lmdeploy_value = np.fromfile(lmdeploy_value_path, dtype=np.float32, count=78 * 8 * 128)
    megatron_value = np.fromfile(megatron_value_path, dtype=np.float32, count=78 * 8 * 128)
    
    lmdeploy_key = lmdeploy_key.reshape([8, 78, 128]).transpose([1, 0, 2])
    megatron_key = megatron_key.reshape([78, 8, 128])
    
    lmdeploy_value = lmdeploy_value.reshape([8, 78, 128]).transpose([1, 0, 2])
    megatron_value = megatron_value.reshape([78, 8, 128])
    
    import pdb;pdb.set_trace()

if __name__ == "__main__":
    # main()
    # test_cache()
    # test_attention()
    # test()
    
    for i in range(40):
        lmdeploy_input = np.fromfile("/data/yocto_bak/analyse/lmdeploy/0108_layer_{}_input.bin".format(i), dtype=np.float32, count=78 * 6144)
        megatron_input = np.fromfile("/data/yocto_bak/analyse/megatron/0108_layer_{}_input.bin".format(i), dtype=np.float32, count=78 * 6144)
        
        print("layer {} input check:".format(i))
        
        print(check(lmdeploy_input, megatron_input))
        
        lmdeploy_output = np.fromfile("/data/yocto_bak/analyse/lmdeploy/0108_layer_{}_output.bin".format(i), dtype=np.float32, count=78 * 6144)
        megatron_output = np.fromfile("/data/yocto_bak/analyse/megatron/0108_layer_{}_output.bin".format(i), dtype=np.float32, count=78 * 6144)
        
        print("layer {} output check:".format(i))
        
        print(check(lmdeploy_output, megatron_output))
        
                
    lmdeploy_ln_input = np.fromfile("/data/yocto_bak/analyse/lmdeploy/final_layernorm_input.bin", dtype=np.float32, count=78 * 6144)
    megatron_ln_input = np.fromfile("/data/yocto_bak/analyse/megatron/final_layernorm_input.bin", dtype=np.float32, count=78 * 6144)
    
    print("final_layernorm_input check:".format(i))
    
    print(check(lmdeploy_ln_input, megatron_ln_input))
        
                
    lmdeploy_ln_output = np.fromfile("/data/yocto_bak/analyse/lmdeploy/final_layernorm_output.bin", dtype=np.float32, count=78 * 6144)
    megatron_ln_output = np.fromfile("/data/yocto_bak/analyse/megatron/final_layernorm_output.bin", dtype=np.float32, count=78 * 6144)
    
    print("final_layernorm_input check:".format(i))
    
    print(check(lmdeploy_ln_output, megatron_ln_output))
        
    lmdeploy_key_cahce = np.fromfile("/data/yocto_bak/analyse/lmdeploy/1109_key_cache.bin", dtype=np.float16, count=8192 * 40 * 8 * 128)
    megatron_key_cahce = np.fromfile("/data/yocto_bak/analyse/megatron/1109_key_cache.bin", dtype=np.float32, count=78 * 40 * 8 * 128)
    
    lmdeploy_key_cahce = (lmdeploy_key_cahce.reshape([40, 8, 8192, 128])[:, :, :78, :]).transpose([0, 2, 1, 3])
    megatron_key_cahce = megatron_key_cahce.reshape([40, 78, 8, 128])
    
    print("key cache check:")
    
    print(check(lmdeploy_key_cahce, megatron_key_cahce))
                
    lmdeploy_value_cahce = np.fromfile("/data/yocto_bak/analyse/lmdeploy/1109_value_cache.bin", dtype=np.float16, count=8192 * 40 * 8 * 128)
    megatron_value_cahce = np.fromfile("/data/yocto_bak/analyse/megatron/1109_value_cache.bin", dtype=np.float32, count=78 * 40 * 8 * 128)
    
    lmdeploy_value_cahce = (lmdeploy_value_cahce.reshape([40, 8, 8192, 128])[:, :, :78, :]).transpose([0, 2, 1, 3])
    megatron_value_cahce = megatron_value_cahce.reshape([40, 78, 8, 128])
    
    print("value cache check:")
    
    print(check(lmdeploy_value_cahce, megatron_value_cahce))
    
    
    lmdeploy_query = np.fromfile("/data/yocto_bak/analyse/lmdeploy/1109_layer_0_query.bin", dtype=np.float32, count=1 * 6 * 8 * 128)
    lmdeploy_query_bias = np.fromfile("/data/yocto_bak/analyse/lmdeploy/1109_layer_0_query_bias.bin", dtype=np.float32, count=1 * 6 * 8 * 128)
    megatron_query = np.fromfile("/data/yocto_bak/analyse/megatron/1109_layer_0_query.bin", dtype=np.float32, count=1 * 6 * 8 * 128)
    
    lmdeploy_key = np.fromfile("/data/yocto_bak/analyse/lmdeploy/1109_layer_0_key.bin", dtype=np.float32, count=1 * 1 * 8 * 128)
    lmdeploy_key_bias = np.fromfile("/data/yocto_bak/analyse/lmdeploy/1109_layer_0_key_bias.bin", dtype=np.float32, count=1 * 1 * 8 * 128)
    megatron_key = np.fromfile("/data/yocto_bak/analyse/megatron/1109_layer_0_key.bin", dtype=np.float32, count=1 * 1 * 8 * 128)
    
    lmdeploy_value = np.fromfile("/data/yocto_bak/analyse/lmdeploy/1109_layer_0_value.bin", dtype=np.float32, count=1 * 1 * 8 * 128)
    lmdeploy_value_bias = np.fromfile("/data/yocto_bak/analyse/lmdeploy/1109_layer_0_value_bias.bin", dtype=np.float32, count=1 * 1 * 8 * 128)
    megatron_value = np.fromfile("/data/yocto_bak/analyse/megatron/1109_layer_0_value.bin", dtype=np.float32, count=1 * 1 * 8 * 128)
    
    lmdeploy_query = lmdeploy_query + lmdeploy_query_bias
    # lmdeploy_query = lmdeploy_query.reshape([6, 8, 128]).transpose([1, 0, 2])
    lmdeploy_query = lmdeploy_query.reshape([8, 6, 128])
    megatron_query = megatron_query.reshape([8, 6, 128])
    
    print("query check:")
    
    print(check(lmdeploy_query, megatron_query))
    
    lmdeploy_key = lmdeploy_key + lmdeploy_key_bias
    lmdeploy_key = lmdeploy_key.reshape([8, 1, 128])
    megatron_key = megatron_key.reshape([8, 1, 128])
    
    print("key check:")
    
    print(check(lmdeploy_key, megatron_key))
    
    lmdeploy_value = lmdeploy_value + lmdeploy_value_bias
    lmdeploy_value = lmdeploy_value.reshape([8, 1, 128])
    megatron_value = megatron_value.reshape([8, 1, 128])
    
    print("value check:")
    
    print(check(lmdeploy_value, megatron_value))
    
    
    lmdeploy_attention_output = np.fromfile("/data/yocto_bak/analyse/lmdeploy/1109_layer_0_attention_output.bin", dtype=np.float32, count=1 * 6144)
    megatron_attention_output = np.fromfile("/data/yocto_bak/analyse/megatron/1109_layer_0_attention_output.bin", dtype=np.float32, count=1 * 6144)
    
    print("attention_output check:".format(i))
    
    print(check(lmdeploy_attention_output, megatron_attention_output))
    
    import pdb;pdb.set_trace()