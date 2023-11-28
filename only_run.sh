#!/bin/bash

PYTHON_MODULE="lmdeploy.turbomind.chat_star_coder"
STAR_CODER_WORKSPACE="./star_coder_workspace/"



if [ ! -d "$STAR_CODER_WORKSPACE" ]; then
    echo "Directory $STAR_CODER_WORKSPACE does not exist. Aborting."
    exit 1
fi

python -m $PYTHON_MODULE $STAR_CODER_WORKSPACE

if [ $? -ne 0 ]; then
    echo "Error: Failed to run python module $PYTHON_MODULE"
    exit 1
fi
