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

from model_center import get_args
from diverse_generation import generate

from infer_dataset import INFER_DATASET

def get_tokenizer(args):
    tokenizer = CPM1Tokenizer(args.vocab_file)
    return tokenizer

def get_model(args, vocab_size):
    config = CPM1Config.from_json_file(args.model_config)
    config.vocab_size = vocab_size
    print ("vocab size:%d"%(vocab_size))

    model = CPM1(config)
    # if args.load != None:
    bmp.load(model, args.load)
    # else:
    #     bmp.init_parameters(model)
    return model

def setup_model(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args, tokenizer.vocab_size)
    bmp.synchronize()
    bmp.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmp.synchronize()
    return tokenizer, model

def initialize():
    # get arguments
    args = get_args()
    # init bmp 
    bmp.init_distributed(seed = args.seed, loss_scale_factor = 2, loss_scale_steps = 1024)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args


def main():
    args = initialize()
    tokenizer, model = setup_model(args)

    fout = open("{}.{}".format(args.output_file, bmp.rank()), "w", encoding="utf-8")

    dataset = INFER_DATASET[args.dataset_name](args.input_file, args.max_length)
    total_lines = len(dataset)
    step = (total_lines + bmp.world_size() -1) // bmp.world_size()
    for idx in range(step):
        # print(bmp.world_size())
        data_idx = step * bmp.rank() + idx

        bmp.print_rank(idx)

        text, golden_summary = dataset[data_idx]
        source = '“' + text + '”的摘要是:'
        
        target_span_len = args.span_length
        # 每个instance指定不同的target span长度
        # target_span_len = int(len(instance['source'][0])*0.4*0.7)

        # TODO: support multi-GPUs for varied target span length
        if target_span_len != args.span_length:
            assert bmp.world_size() == 1, "Using multiple GPUs for varied target span length has not been supported!"

        # 指定最短生成长度
        # min_len = min(target_span_len-1, int(len(instance['source'][0])*0.4*0.7))
        min_len = 2 # 确保生成内容不为空

        predict_sentence = ""

        result = generate(model, tokenizer, source, target_span_len, beam=args.beam_size, beam_group=args.beam_group, diverse_penalty=args.diverse_penalty)
        
        for sent in result:
            fout.write(sent + '\n')
        
    fout.close()

if __name__ == "__main__":
    main()
