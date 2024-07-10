set -e
set -x

export DATA_DIR="your_directory"
export MODEL_REPO=deepseek-ai/deepseek-math-7b-rl


DATASET="/path/AIMO-2nd-Solution/finetune_dataset/reward_train.json"


torchrun --standalone --nproc_per_node=8 \
    finetune_rm.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --source_max_len 1024 \
    --target_max_len 1024 \
    --total_max_len 1024 \
    --per_device_train_batch_size 16 \
    --micro_train_batch_size 1 \
    --learning_rate 2e-5 \
    --lr_eta_min 2e-7 \
    --num_train_epochs 2 \
    --dataset $DATASET \
    --dataset_format "orm" \
    --save_strategy epoch \
    --save_total_limit 1 \
    --train_on_every_token \
    --tensor_parallel_size 1 \
    --save_only_model True \
    --save_dir $DATA_DIR/checkpoints/reward_model\
    --resume_from_checkpoint \
    --param_dtype fp32 \
    --optim_dtype fp32