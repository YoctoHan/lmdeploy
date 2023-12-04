#!/bin/bash

# 定义目标工作目录
target_directory="/data/yocto_bak/lmdeploy"

# 更改到目标目录
cd "$target_directory" || exit

# PYTHON_MODULE="lmdeploy.turbomind.sever_fastertransformer_gpt"
PYTHON_MODULE="lmdeploy.turbomind.chat_star_coder"
# STAR_CODER_WORKSPACE="./star_coder_workspace/"
# VOCAB_DIR="/data3/StarCoderBase/"
STAR_CODER_WORKSPACE="./europa_workspace/"
VOCAB_DIR="/data3/aix2_base_v2/"



if [ ! -d "$STAR_CODER_WORKSPACE" ]; then
    echo "Directory $STAR_CODER_WORKSPACE does not exist. Aborting."
    exit 1
fi

python -m $PYTHON_MODULE $STAR_CODER_WORKSPACE $VOCAB_DIR

if [ $? -ne 0 ]; then
    echo "Error: Failed to run python module $PYTHON_MODULE"
    exit 1
fi
