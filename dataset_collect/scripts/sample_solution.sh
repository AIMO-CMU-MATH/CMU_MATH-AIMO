set -e
set -x


export DATA_DIR="dataset"

TEMPERATURE=$1
SAMPLING_NUM=$2
HOST=$3
METHOD=$4
export OUT_DIR=./exp_results/$5


python3 tora_sample.py --input_path $DATA_DIR \
        --output_path $OUT_DIR/"name" \
        --sampling_num $SAMPLING_NUM \
        --max_tokens 1024 \
        --num_threads 2 \
        --method $METHOD \
        --temperature $TEMPERATURE \
        --info_path=$OUT_DIR/AIME_validation.txt \
        --policy_host http://localhost:$HOST \
