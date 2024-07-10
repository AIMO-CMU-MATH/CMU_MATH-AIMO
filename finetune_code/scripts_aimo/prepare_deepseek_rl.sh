set -e
set -x

export DATA_DIR="your_directory"
export MODEL_REPO=deepseek-ai/deepseek-math-7b-base


python scripts/download.py \
    --repo_id $MODEL_REPO \
    --local_dir $DATA_DIR/checkpoints

python scripts/convert_hf_checkpoint.py \
    --checkpoint_dir $DATA_DIR/checkpoints/$MODEL_REPO