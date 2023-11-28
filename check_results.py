import numpy as np

def analyse(item) -> None:
    print(item)
    megatron_file_path = "/data/yocto_bak/analyse/dynamicDecode/"+ item + "_megatron.bin"
    lmdeploy_file_path = "/data/yocto_bak/analyse/dynamicDecode/"+ item + "_lmdeploy.bin"

    # 打开文件并读取数据
    with open(megatron_file_path, 'rb') as f:
        megatron = f.read()
    megatron_array = np.frombuffer(megatron, dtype=np.float32)
    with open(lmdeploy_file_path, 'rb') as f:
        lmdeploy = f.read()
    lmdeploy_array = np.frombuffer(lmdeploy, dtype=np.float32)

    if item == "post_embedding_output":
        # 获取前 10 大元素的下标
        top10_indices_arr1 = np.argsort(megatron_array)[-10:][::-1]
        top10_indices_arr2 = np.argsort(lmdeploy_array)[-10:][::-1]

        import pdb;pdb.set_trace()

        # 比较下标是否一致
        indices_match = np.array_equal(top10_indices_arr1, top10_indices_arr2)
        print(indices_match)

    # 计算绝对误差和相对误差
    absolute_error = np.abs(megatron_array - lmdeploy_array)
    relative_error = absolute_error / np.abs(megatron_array)

    # 如果所有数据的相对误差都小于0.05（即5%），则认为两份数据是相似的
    all_similar = np.all(relative_error < 0.1)
    print(f"The arrays are similar: {all_similar}")

    # 计算平均绝对误差和平均相对误差
    mean_absolute_error = np.mean(absolute_error)
    mean_relative_error = np.mean(relative_error)

    print(f"Average absolute difference: {mean_absolute_error:.4f}")
    print(f"Average relative difference: {mean_relative_error * 100:.3f}%")


def main():
    check_list = [
        "final_layernorm_outputs",
        "post_embedding_input",
        "post_embedding_weight",
        "post_embedding_output",
    ]
    for item in check_list:
        analyse(item)

def test_matmul():
    index = 6

    input_file_path = "/data/yocto_bak/analyse/dynamicDecode/"+ "post_embedding_input" + "_megatron.bin"
    weight_file_path = "/data/yocto_bak/analyse/dynamicDecode/"+ "post_embedding_weight" + "_megatron.bin"

    # 打开文件并读取数据
    with open(input_file_path, 'rb') as f:
        megatron_input = f.read()
    with open(weight_file_path, 'rb') as f:
        megatron_weight = f.read()

    megatron_input = np.frombuffer(megatron_input, dtype=np.float32).reshape([6144])
    megatron_weight = np.frombuffer(megatron_weight, dtype=np.float32).reshape([49152, 6144])

    megatron_weight = megatron_weight[index]
    print(np.sum(megatron_input * megatron_weight))


    input_file_path = "/data/yocto_bak/analyse/dynamicDecode/"+ "post_embedding_input" + "_lmdeploy.bin"
    weight_file_path = "/data/yocto_bak/analyse/dynamicDecode/"+ "post_embedding_weight" + "_lmdeploy.bin"

    # 打开文件并读取数据
    with open(input_file_path, 'rb') as f:
        lmdeploy_input = f.read()
    with open(weight_file_path, 'rb') as f:
        lmdeploy_weight = f.read()

    lmdeploy_input = np.frombuffer(lmdeploy_input, dtype=np.float32).reshape([6144])
    lmdeploy_weight = np.frombuffer(lmdeploy_weight, dtype=np.float32).reshape([49152, 6144])
    lmdeploy_weight = lmdeploy_weight[index]
    print(np.sum(lmdeploy_input * lmdeploy_weight))

def check_logits():
    file_path = "/data/yocto_bak/analyse/dynamicDecode/"+ "post_embedding_output" + "_lmdeploy.bin"
    with open(file_path, 'rb') as f:
        post_embedding_output = f.read()

    post_embedding_output = np.frombuffer(post_embedding_output, dtype=np.float32).reshape([49152])
    
    top_ten_indices = np.argsort(post_embedding_output)[-10:]
    print(top_ten_indices[::-1])

    file_path = "/data/yocto_bak/analyse/dynamicDecode/"+ "post_embedding_output_2" + "_lmdeploy.bin"
    with open(file_path, 'rb') as f:
        post_embedding_output = f.read()

    post_embedding_output = np.frombuffer(post_embedding_output, dtype=np.float32).reshape([49152])
    
    top_ten_indices = np.argsort(post_embedding_output)[-10:]
    print(top_ten_indices[::-1])

    file_path = "/data/yocto_bak/analyse/dynamicDecode/"+ "post_embedding_output_3" + "_lmdeploy.bin"
    with open(file_path, 'rb') as f:
        post_embedding_output = f.read()

    post_embedding_output = np.frombuffer(post_embedding_output, dtype=np.float32).reshape([49152])
    
    top_ten_indices = np.argsort(post_embedding_output)[-10:]
    print(top_ten_indices[::-1])

    file_path = "/data/yocto_bak/analyse/dynamicDecode/"+ "post_embedding_output_4" + "_lmdeploy.bin"
    with open(file_path, 'rb') as f:
        post_embedding_output = f.read()

    post_embedding_output = np.frombuffer(post_embedding_output, dtype=np.float32).reshape([49152])
    
    top_ten_indices = np.argsort(post_embedding_output)[-10:]
    print(top_ten_indices[::-1])

    file_path = "/data/yocto_bak/analyse/dynamicDecode/"+ "post_embedding_output_5" + "_lmdeploy.bin"
    with open(file_path, 'rb') as f:
        post_embedding_output = f.read()

    post_embedding_output = np.frombuffer(post_embedding_output, dtype=np.float32).reshape([49152])
    
    top_ten_indices = np.argsort(post_embedding_output)[-10:]
    print(top_ten_indices[::-1])

    file_path = "/data/yocto_bak/analyse/dynamicDecode/"+ "post_embedding_output_6" + "_lmdeploy.bin"
    with open(file_path, 'rb') as f:
        post_embedding_output = f.read()

    post_embedding_output = np.frombuffer(post_embedding_output, dtype=np.float32).reshape([49152])
    
    top_ten_indices = np.argsort(post_embedding_output)[-10:]
    print(top_ten_indices[::-1])


    import pdb;pdb.set_trace()


if __name__ == "__main__":
    # main()
    # test_matmul()
    check_logits()