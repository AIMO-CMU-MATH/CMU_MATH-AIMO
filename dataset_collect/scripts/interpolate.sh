set -e
set -x

python3 interpolate.py --base_path "base_path" \
    --finetuned_path "finetuned_path"\
    --save_path "save_path"\
    --alpha 0.1
