import numpy as np
import os
from typing import Optional

def compute_mean_absolute_error(file_path1: str, file_path2: str, dtype, num_elements: int) -> Optional[float]:
    # Check if files exist
    if not os.path.isfile(file_path1) or not os.path.isfile(file_path2):
        print(f"One or both files do not exist: {file_path1}, {file_path2}")
        return None

    data1 = np.fromfile(file_path1, dtype=dtype, count=num_elements)
    data2 = np.fromfile(file_path2, dtype=dtype, count=num_elements)

    # import pdb;pdb.set_trace()

    return np.mean(np.abs(data1 - data2))

def construct_file_path(base_path: str, sub_dir: str, item: str) -> str:
    return os.path.join(base_path, sub_dir, f"{item}.bin")

def main():
    check_list_items = [
        ("input_ids", np.int32, 79),
        ("embedding_table", np.float32, 49152 * 6144),
        ("position_embedding_table", np.float32, 8192 * 6144),
        ("encoder_input", np.float32, 79 * 6144),
        ("context_decoder_input", np.float32, 79 * 6144),
        ("layer_0_pre_attention_layer_norm_weight", np.float32, 6144),
        ("layer_0_pre_attention_layer_norm_bias", np.float32, 6144),
        ("layer_0_pre_attention_layer_norm_output", np.float32, 78 * 6144),
        ("layer_0_attention_output", np.float32, 78 * 6144),
    ]

    base_path = "/data/yocto_bak/analyse/"
    for item, dtype, elements_num in check_list_items:
        lmdeploy_path = construct_file_path(base_path, "lmdeploy", item)
        megatron_path = construct_file_path(base_path, "megatron", item)

        mean_absolute_error = compute_mean_absolute_error(lmdeploy_path, megatron_path, dtype, elements_num)

        if mean_absolute_error is not None:
            print(f"Mean absolute error for {item}: {mean_absolute_error}")

if __name__ == "__main__":
    main()
