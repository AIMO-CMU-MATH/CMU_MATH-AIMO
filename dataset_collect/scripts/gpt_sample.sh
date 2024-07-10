set -e
set -x


python3 gpt_access.py --input_path "dataset_path" \
        --output_path "output_path" \
        --max_sampling_num 64 \
        --correct_sampling_num 32