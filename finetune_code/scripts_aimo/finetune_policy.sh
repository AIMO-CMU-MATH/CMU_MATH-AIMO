set -e
set -x


export DATA_DIR="your_directory"
export MODEL_REPO=deepseek-ai/deepseek-math-7b-rl
export OMP_NUM_THREADS=8

torchrun --standalone --nproc_per_node=8 \
    finetune.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --source_max_len 1024 \
    --target_max_len 1024 \
    --total_max_len 1024 \
    --per_device_train_batch_size 16 \
    --micro_train_batch_size 1 \
    --learning_rate 2e-5 \
    --lr_eta_min 2e-7 \
    --num_train_epochs 3 \
    --dataset "/path/AIMO-2nd-Solution/finetune_dataset/policy_train.json" \
    --dataset_format "aimo" \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_dir $DATA_DIR/checkpoints/policy_model \
    --save_only_model True \
    --tensor_parallel_size 1 \
    --weight_decay 0.1 \
    --warmup_ratio 0.02 \
    --param_dtype fp32 \
    --optim_dtype fp32