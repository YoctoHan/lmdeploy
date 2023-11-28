#!/bin/bash

WORK_DIR="/data/yocto_bak/lmdeploy"
BUILD_DIR="build"
# PYTHON_MODULE="lmdeploy.turbomind.chat_star_coder"
PYTHON_MODULE="lmdeploy.turbomind.sever_fastertransformer_gpt"
STAR_CODER_WORKSPACE="./star_coder_workspace/"

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

proxychains make -j128

if [ $? -ne 0 ]; then
    echo "Error: make command failed"
    exit 1
fi

cd -

python -m $PYTHON_MODULE $STAR_CODER_WORKSPACE

if [ $? -ne 0 ]; then
    echo "Error: Failed to run python module $PYTHON_MODULE"
    exit 1
fi
