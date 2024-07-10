set -e
set -x
#!/bin/bash
DEVICE=$1
PORT=$2


MODEL_REPO="model_path"
CUDA_VISIBLE_DEVICES=$DEVICE python3 -m sglang.launch_server --model-path $MODEL_REPO --port $PORT --tp-size 1
