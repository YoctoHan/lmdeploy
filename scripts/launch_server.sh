#!/bin/bash

# # 启用调试模式
# set -x

# 定义目标工作目录
target_directory="/data/yocto_bak/lmdeploy"

# 更改到目标目录
cd "$target_directory" || exit

WORK_DIR="/data/yocto_bak/lmdeploy"
BUILD_DIR="build"
PYTHON_MODULE="lmdeploy.turbomind.server_europa"
# PYTHON_MODULE="lmdeploy.turbomind.chat_star_coder"
# STAR_CODER_WORKSPACE="./star_coder_workspace/"
# VOCAB_DIR="/data3/StarCoderBase/"
STAR_CODER_WORKSPACE="./europa_workspace/"
VOCAB_DIR="/data3/aix2_base_v2/"
IS_INSTRUCT_MODEL="False"
ATTENTION_HEAD_TYPE="groupedquery"
PORT="12316"

cd $WORK_DIR

if [ $? -ne 0 ]; then
    echo "Error: Failed to change to directory $WORK_DIR"
    exit 1
fi

cd $BUILD_DIR

if [ $? -ne 0 ]; then
    echo "Error: Failed to change to directory $BUILD_DIR"
    exit 1
fi

proxychains make -j56

if [ $? -ne 0 ]; then
    echo "Error: make command failed"
    exit 1
fi

cd -

python -m $PYTHON_MODULE main \
                         --model_dir=${STAR_CODER_WORKSPACE} \
                         --vocab_dir=${VOCAB_DIR} \
                         --is_instruct_model=${IS_INSTRUCT_MODEL} \
                         --attention_head_type=${ATTENTION_HEAD_TYPE} \
                         --port=${PORT}

if [ $? -ne 0 ]; then
    echo "Error: Failed to run python module $PYTHON_MODULE"
    exit 1
fi

# # 关闭调试模式
# set +x 
