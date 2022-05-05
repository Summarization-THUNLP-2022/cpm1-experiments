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

from model_center import get_args
from generation import generate

from infer_dataset import INFER_DATASET, BatchInferDataset

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
    total_lines = dataset.total_length
    step = (total_lines + bmp.world_size() -1) // bmp.world_size()
    dataset.read_dataset(step * bmp.rank(), step * (bmp.rank() + 1))
    batch_num = (step + args.batch_size - 1) // args.batch_size
    batch_dataset = BatchInferDataset(dataset, tokenizer, args.span_length, args.batch_size, batch_num)
    min_len = 2 # 确保生成内容不为空
    def work(input_dict):
        result = generate(model, tokenizer, input_dict, beam=args.beam_size,
                            temperature = args.temperature, top_k = args.top_k, top_p = args.top_p,
                            no_repeat_ngram_size = args.no_repeat_ngram_size, repetition_penalty = args.repetition_penalty, 
                            random_sample=args.random_sample, min_len=min_len)
        
        for sent in result:
            fout.write(sent + '\n')
            fout.flush()

    if bmp.rank() == 0:
        for input_dict in tqdm(batch_dataset):
            work(input_dict)
    else:
        for input_dict in batch_dataset:
            work(input_dict)
        
    fout.close()

if __name__ == "__main__":
    main()
