import torch
import bmtrain as bmt
import os

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
    for split in splits:
        dataset[split] = DATASET[dataset_name](base_path, split, rank, world_size, tokenizer, args.max_length)
    return dataset


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

            optimizer.zero_grad()

            logits = model(input_tokens, input_length, input_context, input_span)
            # bmt.print_rank(logits[0])

            loss = loss_func(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # 计算每个字符的平均loss
            total_target_length = torch.sum(target_length)
            global_loss = loss * total_target_length
            global_loss = bmt.sum_loss(global_loss, method='sum').item()
            global_length = bmt.sum_loss(total_target_length, method='sum').item()
            avg_loss_now = global_loss / global_length

            loss = optimizer.loss_scale(loss)
            loss.backward()
            grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups, args.clip_grad, scale = optimizer.scale, norm_type = 2)

            bmt.optim_step(optimizer, lr_scheduler)

            bmt.print_rank(
                "train | epoch {:3d} | Iter: {:6d}/{:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f} |".format(
                    epoch,
                    it,
                    len(dataloader["train"]),
                    avg_loss_now,
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
