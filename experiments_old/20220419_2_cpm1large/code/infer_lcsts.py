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
from generation import generate

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
    fin = open('{}'.format(args.input_file), 'r', encoding='utf-8')
    lines = fin.readlines()
    fin.close()
    total_lines = len(lines)
    step = (total_lines + bmp.world_size() -1) // bmp.world_size()
    for idx in range(step):
        # print(bmp.world_size())
        data_idx = step * bmp.rank() + idx
        print(idx, bmp.rank())

        if data_idx >= total_lines:
            source = '“' + json.loads(lines[0])['text'] + '”的摘要是:'
        else:
            source = '“' + json.loads(lines[data_idx])['text'] + '”的摘要是:'
        
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

        for it in generate(model, tokenizer, source, target_span_len, beam=args.beam_size,
                            temperature = args.temperature, top_k = args.top_k, top_p = args.top_p,
                            no_repeat_ngram_size = args.no_repeat_ngram_size, repetition_penalty = args.repetition_penalty, 
                            random_sample=args.random_sample, min_len=min_len):
            
            if it == '<eod>':
                break

            predict_sentence += it
            # fout.write(it)
            # fout.flush()

        result_dict = {
            "summary": predict_sentence,
            "text": json.loads(lines[data_idx])['text']
        }

        bmp.synchronize()

        if data_idx >= total_lines:
            continue

        fout.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
        fout.flush()
        
    fout.close()

if __name__ == "__main__":
    main()
