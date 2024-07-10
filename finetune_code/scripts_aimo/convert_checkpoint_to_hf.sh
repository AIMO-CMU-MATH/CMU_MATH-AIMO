set -x
set -e

MODEL_NAME=$1
export DATA_DIR="your_directory"

python -u convert_checkpoint_to_hf.py \
    --tp_ckpt_name $DATA_DIR/checkpoints/${MODEL_NAME} \
    --pretrain_name $DATA_DIR/checkpoints/deepseek-ai/deepseek-math-7b-rl \
    --tokenizer_name "deepseek-ai/deepseek-math-7b-rl" \
    --save_name_hf $DATA_DIR/${MODEL_NAME}
