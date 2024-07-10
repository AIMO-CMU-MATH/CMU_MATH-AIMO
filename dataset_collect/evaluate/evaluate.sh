set -e 
set -x

export path=/nobackup/users/zhiqings/yangzhen/project/tree_search/exp_results/pal_sample/T_1-samp_num_32-sft_inter_0.3
 

python3 ../math_evaluate.py   --path $path.json \
    --agg_func majority_vote \
    --model_type deepseek_interpolated \
    --output_path $path.txt

