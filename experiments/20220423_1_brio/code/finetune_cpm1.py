import torch
import bmtrain as bmt
import os
import torch.nn.functional as F

from model_center import get_args
from model_center.model import CPM1
from model_center.tokenizer import CPM1Tokenizer
# from model_center.dataset.cpm1dataset import DATASET
from cpm1_dataset import DATASET
from model_center.utils import print_inspect
from model_center.dataset import DistributedDataLoader

def get_tokenizer(args):
    tokenizer = CPM1Tokenizer.from_pretrained(args.model_config, cache_path=args.cache_path)
    return tokenizer

def get_model(args):
    model = CPM1.from_pretrained(args.model_config, cache_path=args.cache_path)
    return model

def get_optimizer(args, model):
    optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters(), 
                                               weight_decay=args.weight_decay, 
                                               scale=args.loss_scale)
    return optimizer

def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    if args.lr_decay_style == "noam":
        lr_scheduler = bmt.lr_scheduler.Noam(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "constant":
        lr_scheduler = bmt.lr_scheduler.NoDecay(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = -1,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "linear":
        lr_scheduler = bmt.lr_scheduler.Linear(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "exponential":
        lr_scheduler = bmt.lr_scheduler.Exponential(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = bmt.lr_scheduler.Cosine(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler

def setup_model_and_optimizer(args):
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    # get the model
    model = get_model(args)
    bmt.synchronize()
    # get the optimizer and lr_scheduler
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    # get the memory usage
    bmt.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmt.synchronize()
    return tokenizer, model, optimizer, lr_scheduler

def initialize():
    # get arguments
    args = get_args()
    # init bmt 
    bmt.init_distributed(seed = args.seed, loss_scale_factor = 2, loss_scale_steps = 1024)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args


def prepare_dataset(args, tokenizer, base_path, dataset_name, rank, world_size):
    splits = ['train', 'dev']
    dataset = {}
    dataset['train'] = DATASET[dataset_name + '_train'](base_path, split, rank, world_size, tokenizer, args.max_length)
    dataset['dev'] = DATASET[dataset_name](base_path, split, rank, world_size, tokenizer, args.max_length)
    return dataset


def RankingLoss(score, summary_score=None, margin=0, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
    ones = torch.ones_like(score)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)
    # candidate loss
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss


def finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset):
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    dataloader = {
        "train": DistributedDataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True),
        "dev": DistributedDataLoader(dataset['dev'], batch_size=args.batch_size, shuffle=False),
    }

    for epoch in range(5):
        model.train()
        for it, data in enumerate(dataloader['train']):
            input_tokens = data["input_tokens"].cuda()
            input_length = data["input_length"].cuda()
            input_context = data["input_context"].cuda()
            input_span = data["input_span"].cuda()
            targets = data["targets"].cuda()
            target_length = data["target_length"].cuda()

            batch_size = input_tokens.size(0)
            cand_num = input_tokens.size(1)

            input_tokens = input_tokens.view(batch_size * cand_num, input_tokens.size(-1))
            input_length = input_length.view(batch_size * cand_num)
            input_context = input_context.view(batch_size * cand_num, input_context.size(-1))
            input_span = input_span.view(batch_size * cand_num, input_span.size(-1))

            optimizer.zero_grad()

            logits = model(input_tokens, input_length, input_context, input_span)

            logits = logits.view(batch_size, cand_num, logits.size(-2), logits.size(-1))
            probs = F.log_softmax(logits, dim=3)
            scores = torch.gather(probs, 3, targets.unsqueeze(-1)).squeeze(-1)
            target_mask = (targets != -100)
            scores = torch.mul(scores, target_mask).sum(-1) / (target_mask.sum(-1) ** args.length_penalty) # [bz, cand_num]
            if args.no_gold:
                ranking_loss = RankingLoss(scores[:, 1:], margin=args.margin, no_gold=True)
            else:
                ranking_loss = RankingLoss(scores[:, 1:], scores[:, 0], args.margin, args.gold_margin, args.gold_weight)
            
            mle_probs = logits[:, 0]
            mle_targets = targets[:, 0]

            mle_loss = loss_func(mle_probs.view(-1, mle_probs.size(-1)), mle_targets.view(-1))

            loss = args.rank_weight * ranking_loss + args.mle_weight * mle_loss
            
            global_ranking_loss = bmt.sum_loss(ranking_loss).item()
            global_mle_loss = bmt.sum_loss(mle_loss).item()
            global_loss = bmt.sum_loss(global_loss).item()

            loss = optimizer.loss_scale(loss)
            loss.backward()
            grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups, args.clip_grad, scale = optimizer.scale, norm_type = 2)

            bmt.optim_step(optimizer, lr_scheduler)

            bmt.print_rank(
                "train | epoch {:3d} | Iter: {:6d}/{:6d} | ranking_loss: {:.4f} | mle_loss: {:.4f} |loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f} |".format(
                    epoch,
                    it,
                    len(dataloader["train"]),
                    global_ranking_loss,
                    global_mle_loss,
                    global_loss,
                    lr_scheduler.current_lr,
                    int(optimizer.scale),
                    grad_norm
                )
            )
            # if it % args.inspect_iters == 0: print_inspect(model, "*")
            if args.save != None and it % args.save_iters == 0:
                bmt.save(model, os.path.join(args.save, args.save_name + f"-{epoch}-{it}.pt"))

        model.eval()
        with torch.no_grad():
            avg_loss = 0
            total = 0
            for it, data in enumerate(dataloader['dev']):
                input_tokens = data["input_tokens"].cuda()
                input_length = data["input_length"].cuda()
                input_context = data["input_context"].cuda()
                input_span = data["input_span"].cuda()
                targets = data["targets"].cuda()
                target_length = data["target_length"].cuda()

                logits = model(input_tokens, input_length, input_context, input_span)
                loss = loss_func(logits.view(-1, logits.size(-1)), targets.view(-1))

                # 计算每个字符的平均loss以及总字符数量
                total_target_length = torch.sum(target_length)
                loss *= total_target_length
                global_loss = bmt.sum_loss(loss, method='sum').item()
                global_length = bmt.sum_loss(total_target_length, method='sum').item()
                avg_loss_now = global_loss / global_length
                total += global_length
                avg_loss += (avg_loss_now - avg_loss) * global_length / total

                bmt.print_rank(
                    "dev | epoch {:3d} | Iter: {:6d}/{:6d} | loss: {:.4f} |".format(
                        epoch,
                        it,
                        len(dataloader["dev"]),
                        avg_loss_now
                    )
                )
            
            bmt.print_rank(f"dev epoch {epoch}: avg_loss: {avg_loss}")

def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    dataset = prepare_dataset(
        args,
        tokenizer,
        args.data_path,
        args.dataset_name,
        bmt.rank(), bmt.world_size(),
    )
    finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset)

if __name__ == "__main__":
    main()
