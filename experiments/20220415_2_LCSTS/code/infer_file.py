#coding:utf-8

import time
import random
import torch
import bmtrain as bmp
import numpy as np
import os
import json

from model_center.model import CPM1
from model_center.tokenizer import CPM1Tokenizer
from model_center import get_args
from generation import generate_no_beam_cpm3, generate


def get_tokenizer(args):
    tokenizer = CPM1Tokenizer.from_pretrained(args.model_config, cache_path=args.cache_path)
    return tokenizer

def get_model(args):
    model = CPM1.from_pretrained(args.model_config, cache_path=args.cache_path)
    return model

def setup_model(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args)
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

# def get_ppl(sentA : str, results : List[str], tokenizer : T5Tokenizer, model : T5):
#     with torch.inference_mode():
#         enc_tensor, enc_len = make_input( tokenize(tokenizer, sentA) )
#         dec_input = []
#         dec_target = []
#         for i, r in enumerate(results):
#             tokens = tokenizer.encode(r)
#             span_idx = tokenizer.get_span(i)
#             dec_input.append( span_idx )
#             dec_target.append( span_idx )
#             dec_input.extend( tokens )
#             dec_target.extend( tokens )
        
#         dec_target.append( tokenizer.eod_id )
#         dec_target = dec_target[1:]

        
#         dec_tensor, dec_len = make_input(dec_input)
#         while len(dec_target) < dec_tensor.size(1):
#             dec_target.append(-100)
#         target_tensor = torch.tensor([dec_target]).long().cuda()
    
#         logits = model(enc_tensor, enc_len, dec_tensor, dec_len)
#         loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
#         batch, seq_len, vocab_out_size = logits.size()
#         loss = loss_func(logits.view(batch * seq_len, vocab_out_size), target_tensor.view(batch * seq_len))
#         loss = loss.cpu().item()
#         print(enc_tensor.size(), dec_tensor.size())
#         print("Loss: %lf" % loss)
#         print("PPL: %lf" % math.exp(loss))

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
        # print(idx, bmp.rank())
        # print(bmp.world_size())
        data_idx = step * bmp.rank() + idx

        if data_idx >= total_lines:
            instance = {
                'mode': 'lm',
                'source': ["空"],
                'target': "空",
                'control': {
                    'keywords': [],
                    'entities': [],
                    'genre': '',
                    'relations': [],
                    'events': []
                }
            }
        else:
            instance = json.loads(lines[data_idx])

        if instance['mode'] == 'lm':
            eos_max = 1
        else:
            eos_max = 2
        
        target_span_len = args.span_length
        # 每个instance指定不同的target span长度
        # target_span_len = int(len(instance['source'][0])*0.4*0.7)

        # TODO: support multi-GPUs for varied target span length
        if target_span_len != args.span_length:
            assert bmp.world_size() == 1, "Using multiple GPUs for varied target span length has not been supported!"

        eos_num = 0
        # 指定最短生成长度
        # min_len = min(target_span_len-1, int(len(instance['source'][0])*0.4*0.7))
        min_len = None
        for it in generate(model, tokenizer, instance, target_span_len, beam=args.beam_size,
                            temperature = args.temperature, top_k = args.top_k, top_p = args.top_p,
                            no_repeat_ngram_size = args.no_repeat_ngram_size, repetition_penalty = args.repetition_penalty, 
                            random_sample=args.random_sample, min_len=min_len):
            
            if eos_num == eos_max:
                break
            
            if it == '</s>':
                eos_num += 1
            
            fout.write(it)
            fout.flush()
        fout.write('\n')
        
    fout.close()

if __name__ == "__main__":
    main()
