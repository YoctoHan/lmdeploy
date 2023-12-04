#!/bin/bash

# 定义目标工作目录
target_directory="/data/yocto_bak/lmdeploy"

# 更改到目标目录
cd "$target_directory" || exit

# 在这里定义参数
model_name="AixEuropaBaseV2"
model_path="/data3/aix2_base_v2/"
model_format="europa"
tokenizer_path="/data3/aix2_base_v2/aixTokenizer"
dst_path="./europa_workspace"

# 显示参数
echo "Deploy Parameter:"
echo "model_name: $model_name"
echo "model_path: $model_path"
echo "model_format: $model_format"
echo "tokenizer_path: $tokenizer_path"
echo "dst_path: $dst_path"

# 运行 Python 命令
echo
echo "Deploying $model_name..."
python3 -m lmdeploy.serve.turbomind.deploy "$model_name" "$model_path" "$model_format" "$tokenizer_path" "$dst_path"

echo
echo "DONE."