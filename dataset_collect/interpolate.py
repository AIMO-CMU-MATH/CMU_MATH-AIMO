from transformers import AutoModelForCausalLM
import torch
import argparse

def main(args):
    base_model = AutoModelForCausalLM.from_pretrained(args.base_path)
    finetuned_model = AutoModelForCausalLM.from_pretrained(args.finetuned_path)
    interpolated_model = AutoModelForCausalLM.from_pretrained(args.finetuned_path, config=base_model.config)
    for name1, param1 in base_model.named_parameters():
        param2 = finetuned_model.get_parameter(name1)
        param_new = interpolated_model.get_parameter(name1)
        if "embed" in name1 or "lm_head" in name1:
            param_new.data.copy_(param2.data)
            print(name1)
        else:
            param_new.data.copy_((1 - args.alpha) * param1.data + args.alpha * param2.data)
    interpolated_model.to(dtype=torch.bfloat16)
    interpolated_model.save_pretrained(args.save_path)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--base_path', type=str, required=True)
    args_parser.add_argument('--finetuned_path', type=str, required=True)
    args_parser.add_argument('--save_path', type=str, required=True)
    args_parser.add_argument('--alpha', type=float, default=1.0)
    args = args_parser.parse_args()
    main(args)