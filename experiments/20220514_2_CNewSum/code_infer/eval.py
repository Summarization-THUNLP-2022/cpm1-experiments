#coding:utf-8

import torch
import numpy as np
import os
from model_center.model import CPM1Config,CPM1 
from model_center.tokenizer import CPM1Tokenizer 
from tqdm import tqdm
import torch.distributed as dist
import torch.nn.functional as F

from model_center import get_args
from generation import generate

from eval_dataset import CNewSumEvalDataset

def get_tokenizer(args):
    tokenizer = CPM1Tokenizer(args.vocab_file)
    return tokenizer

def get_model(args, vocab_size):
    config = CPM1Config.from_json_file(args.model_config)
    config.vocab_size = vocab_size
    print ("vocab size:%d"%(vocab_size))

    model = CPM1(config).cuda()
    # if args.load != None:
    model.load_state_dict(
        torch.load(args.load),
        strict = True
    )
    torch.cuda.synchronize()
    return model

def setup_model(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args, tokenizer.vocab_size)
    return tokenizer, model

def initialize():
    # get arguments
    args = get_args()
    # init bmp 
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group("nccl")

    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args


def main():
    args = initialize()
    tokenizer, model = setup_model(args)

    fout = open("{}.{}".format(args.output_file, args.local_rank), "w", encoding="utf-8")

    dataset = CNewSumEvalDataset(tokenizer, args.input_file, args.candidate_file, args.candidate_num)
    dataset.read_dataset()
    total_lines = len(dataset)
    step = (total_lines + dist.get_world_size() -1) // dist.get_world_size()
    dataset.read_dataset(step * args.local_rank, step * (args.local_rank + 1))
    def work(data):
        input_tokens = data["input_tokens"].cuda()
        input_length = data["input_length"].cuda()
        input_context = data["input_context"].cuda()
        input_span = data["input_span"].cuda()
        targets = data["targets"].cuda()
        target_length = data["target_length"].cuda()
        
        logits, past_key_values = model(input_tokens, input_length, input_context, input_span)
        probs = F.log_softmax(logits, dim=2)
        target_mask = (targets != -100)
        scores = torch.gather(probs, 2, torch.mul(targets, target_mask).long().unsqueeze(-1)).squeeze(-1)
        scores = torch.mul(scores, target_mask).sum(-1) / (target_mask.sum(-1) ** args.length_penalty) # [bz]
        best_idx = torch.argmax(scores)
        return best_idx


    with torch.inference_mode():
        if args.local_rank == 0:
            for idx, input_dict in tqdm(enumerate(dataset)):
                best_idx = work(input_dict)
                fout.write(dataset._data[idx]['candidate'][best_idx] + '\n')
                fout.flush()
        else:
            for idx, input_dict in enumerate(dataset):
                best_idx = work(input_dict)
                fout.write(dataset._data[idx]['candidate'][best_idx] + '\n')
                fout.flush()
        
    fout.close()

if __name__ == "__main__":
    main()
