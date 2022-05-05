#coding:utf-8

import time
import random
import torch
import bmtrain as bmp
import numpy as np
import os
import json
from model_center.model import CPM1Config,CPM1 
from model_center.tokenizer import CPM1Tokenizer 
from tqdm import tqdm
import torch.distributed as dist
from model_center import get_args
from diverse_generation import diverse_beam_search_generate

from infer_dataset import INFER_DATASET, BatchInferDataset

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

    dataset = INFER_DATASET[args.dataset_name](args.input_file, args.max_length)
    total_lines = dataset.total_length
    step = (total_lines + dist.get_world_size() -1) // dist.get_world_size()
    dataset.read_dataset(step * args.local_rank, step * (args.local_rank + 1), tokenizer)
    batch_num = (step + args.batch_size - 1) // args.batch_size
    batch_dataset = BatchInferDataset(dataset, tokenizer, args.span_length, args.batch_size, batch_num)
    min_len = 2 # 确保生成内容不为空
    def work(global_step, input_dict):
        result = diverse_beam_search_generate(model, tokenizer, input_dict, beam_size=args.beam_size, 
                                              beam_group=args.beam_group, diverse_penalty=args.diverse_penalty, 
                                              no_repeat_ngram_size = args.no_repeat_ngram_size, 
                                              repetition_penalty = args.repetition_penalty, min_len=min_len)
        
        for idx, sent in enumerate(result):
            if global_step * args.batch_size + idx // args.beam_size >= len(dataset):
                continue
            fout.write(sent + '\t' + str(input_dict['ids'][idx // args.beam_size]) + '\n')
            fout.flush()

    if args.local_rank == 0:
        for global_step, input_dict in tqdm(enumerate(batch_dataset)):
            work(global_step, input_dict)
    else:
        for global_step, input_dict in enumerate(batch_dataset):
            work(global_step, input_dict)
        
    fout.close()


if __name__ == "__main__":
    main()
