import torch
import torch.nn.functional as F
import numpy as np
import bmtrain as bmp
import math

class BeamHypotheses(object):

    def __init__(self, n_hyp, max_len, length_penalty, early_stopping, tokenizer=None):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        # try to penalize repetation (fail)
        # score = sum_logprobs / len(set(hyp.cpu().tolist())) ** self.length_penalty
        
        # bmp.print_rank(sum_logprobs, len(hyp))
        # bmp.print_rank(f'score = {score}, hyp = {self.tokenizer.decode(hyp.cpu().tolist())}')
        # bmp.print_rank('============================')

        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)
        
    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / cur_len ** self.length_penalty

def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    # 拿到n-gram中需要ban的最后一个token的list
    # return banned_ngrams.get(ngram_idx, []), ngram_idx # for debug
    return banned_ngrams.get(ngram_idx, [])


def calc_banned_ngram_tokens(
    prev_input_ids: torch.Tensor, num_hypos: int, ngram_size: int, start_idx=None, end_idx=None, window_size=None, tokenizer=None):
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if start_idx is not None and end_idx is not None:
        # 可能end_idx < start_idx，但符合逻辑
        if window_size:
            prev_input_ids = prev_input_ids[:, max(start_idx, end_idx + 1 - window_size): end_idx+1]
        else:
            prev_input_ids = prev_input_ids[:, start_idx: end_idx+1]
        
    cur_len = prev_input_ids.size(1)
    
    if cur_len + 1 < ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]

    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)

    banned_tokens = [
        _get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
        for hypo_idx in range(num_hypos)
    ]
    # for hypo_idx in range(num_hypos):
    #     bmp.print_rank(tokenizer.decode(list(banned_tokens[hypo_idx][1])) + "|" + "/".join([tokenizer.decode([x]) for x in banned_tokens[hypo_idx][0]]))
    return banned_tokens
    # return [x[0] for x in banned_tokens]

# min_length_constriant
def min_length_constraint(logits, cur_len, min_len, tokenizer):
    # This enforcing a min-length by setting EOS probability to 0.
    if cur_len < min_len:
        logits[:, tokenizer.eos_id] = -float("inf")


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float("inf")):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    batch_size = logits.size()[0]
    if top_p > 0.0:
        logits=logits.view(batch_size, -1).contiguous()
        for index in range(len(logits)):

            sorted_logits, sorted_indices = torch.sort(logits[index].view(-1), descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[index][indices_to_remove] = filter_value

        logits=logits.view(batch_size, -1).contiguous()

    return logits


def calc_banned_bad_words_ids(prev_input_ids, bad_words_ids, start_idx=None,  end_idx=None):
    if start_idx is not None and end_idx is not None:
        # 可能end_idx < start_idx，但符合逻辑
        prev_input_ids = prev_input_ids[:, start_idx: end_idx+1]
        
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue
            # 如果最后一个token之前的token都match上了，那就把最后一个token禁掉
            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens


def enforce_repetition_penalty_(tokenizer, 
                                lprobs, 
                                batch_size, 
                                num_beams, 
                                prev_output_tokens, 
                                repetition_penalty,
                                start_idx=None,
                                end_idx=None,
                                window_size=None):
    # 改为只对output token做惩罚
    assert repetition_penalty >= 1, "repetition penalty coefficient should >= 1"
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    for i in range(batch_size * num_beams):
        if start_idx is None or end_idx is None:
            output_tokens = prev_output_tokens[i].tolist()
        else:
            if end_idx >= start_idx:
                if window_size:
                    output_tokens = prev_output_tokens[i][max(start_idx, end_idx + 1 - window_size): end_idx+1].tolist()
                else:
                    output_tokens = prev_output_tokens[i][start_idx: end_idx+1].tolist()
            else:
                output_tokens = []
        #print(output_tokens)
        for previous_token in set(output_tokens):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty


def postprocess_next_token_scores(tokenizer,
                                  scores,
                                  input_ids,
                                  no_repeat_ngram_size,
                                  bad_words_ids,
                                  repetition_penalty,
                                  batch_size,
                                  num_beams,
                                  start_idx=None,
                                  end_idx=None,
                                  window_size=None,
                                  min_len=None):

    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0:
        enforce_repetition_penalty_(
            tokenizer, scores, batch_size, num_beams, input_ids, repetition_penalty, start_idx, end_idx, window_size
        )

    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = calc_banned_ngram_tokens(input_ids, num_batch_hypotheses, no_repeat_ngram_size, start_idx, end_idx, window_size, tokenizer)
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    if bad_words_ids is not None:
        # calculate a list of banned tokens according to bad words
        banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids, start_idx, end_idx)

        for i, banned_tokens in enumerate(banned_tokens):
            scores[i, banned_tokens] = -float("inf")

    # 允许生成eos和bos，以及换行
    scores[:, [0, 1, 2, 3] + [5] + [x for x in range(8, 20)]] = -float("inf")

    if start_idx is not None and end_idx is not None and end_idx >= start_idx and min_len is not None:
        min_length_constraint(scores, end_idx - start_idx + 1, min_len, tokenizer)

    return scores


def round_up(x, d):
    return (x + d - 1) // d * d

def make_input_cpm3(ctx, info, prompt_length):
    task = info[0]
    len_ctx = len(ctx)
    inp = np.arange((prompt_length+len_ctx), dtype = np.int64) + prompt_length * task
    inp[prompt_length:] = ctx[:len_ctx]
    len_inp = len(inp)

    info = [x + prompt_length for x in info[1:]]
    context_inp = np.full(len_inp, True)
    # 保证end一定能看见
    for i in range(1, len(info)-1, 2):
        context_inp[info[i]:info[i+1]] = False
    
    tgt = np.full((len_inp), -100, dtype = np.int64)
    tgt[:-1] = np.where(
        context_inp[1:],
        -100,
        inp[1:]
    )

    position_inp = np.arange((len_inp), dtype = np.float32) / prompt_length
    segment_inp = np.zeros((len_inp), dtype = np.int64)

    if task == 0:
        arr = [(2, info[0]), (1, 0), (1, info[-1])]
    else:
        arr = [(2, info[0]), (2+task, info[1]), (1, info[-1])]
    
    last = prompt_length
    for (typ, end) in arr:
        if end > last:
            segment_inp[last:end] = typ
            position_inp[last:end] = np.arange(end-last) / (end-last)
            last = end
    assert last == len_inp
    # print("inp:\n", inp)
    # print("tgt:\n", tgt)
    # print("len_input:\n", len_inp)
    # print("context_inp:\n", context_inp)
    # print("position_inp:\n", position_inp)
    # print("segment_inp:\n", segment_inp)

    max_length = round_up(len_inp, 2)

    _ctx = torch.zeros((max_length,), dtype=torch.long)
    _ctx[:len_inp] = torch.from_numpy(inp)[:len_inp].long()
    _context = torch.full((max_length,), False, dtype=torch.bool)
    _context[:len_inp] = torch.from_numpy(context_inp)[:len_inp].bool()
    _position = torch.full((max_length,), False, dtype=torch.float)
    _position[:len_inp] = torch.from_numpy(position_inp)[:len_inp].float()
    _segment = torch.full((max_length,), False, dtype=torch.long)
    _segment[:len_inp] = torch.from_numpy(segment_inp)[:len_inp].long()
    _tgt = torch.full((max_length,), -100, dtype=torch.long)
    _tgt[:len_inp] = torch.from_numpy(tgt)[:len_inp].long()

    _span = torch.zeros((max_length + 1,), dtype=torch.long)
    _span[len_inp] = 1  # 每个拼接的句子结尾的后一位是1
    _span = torch.cumsum(_span, dim=-1)[:-1]

    len_cxt = torch.LongTensor([len_inp])

    return _ctx.unsqueeze(0), len_cxt, _context.unsqueeze(0),\
           _position.unsqueeze(0), _segment.unsqueeze(0), _span.unsqueeze(0), _tgt.unsqueeze(0)

def get_control(control, tokenizer, task):
    sep_id1 = 30665
    sep_id2 = 30666
    keywords = []
    if 'keywords' in control and control['keywords'] != []:
        keywords_set = set()
        for i, keyword in enumerate(control['keywords']):
            if keyword not in keywords_set:
                keywords_set.add(keyword)
                keywords += tokenizer.encode(keyword)
            if i != len(control['keywords']) - 1:
                keywords += [sep_id1]
        keywords = [tokenizer.begin_of_keyword_id] + keywords +[tokenizer.end_of_keyword_id]

    if 'genre' in control and control['genre'] != "" and control['genre'] not in ['书籍', '杂志']:
        style = tokenizer.encode(control['genre'])
        style = [tokenizer.begin_of_style_id] + style +[tokenizer.end_of_style_id]
    else:
        style = []

    relations = []
    if 'relations' in control and control['relations'] != []:
        relation_set = set()
        for items in control['relations']:
            relation_join = "/".join(items)
            if relation_join not in relation_set:
                relation_set.add(relation_join)
                relation = []
                for i, item in enumerate(items):
                    relation += tokenizer.encode(item)
                    if i != len(items) - 1:
                        relation += [sep_id1]
                ids = [tokenizer.begin_of_relation_id] + relation + [tokenizer.end_of_relation_id]
                relations = relations + ids

    events = []
    if 'events' in control and control['events'] != []:
        events_set = set()
        for i in control['events']:
            event_join = "/".join([":".join(x) for x in sorted(i.items(), key=lambda x:x[0])])
            if event_join not in events_set:
                events_set.add(event_join)
                event = []
                for idx, j in enumerate(i):
                    event += tokenizer.encode(j) + [sep_id2] + tokenizer.encode(i[j])
                    if idx != len(i) - 1:
                        event += [sep_id1]
                ids = [tokenizer.begin_of_event_id] + event + [tokenizer.end_of_event_id]
                events = events + ids

    if task == 0:
        res = keywords + relations + events
    elif task == 1:
        res = keywords
    elif task == 2:
        res = style + keywords + relations + events
    elif task == 3:
        res = style
    else:
        res = []
    
    # bmp.print_rank(res)

    return res


def convert_to_ids(tokenizer, text):
    ids = tokenizer.encode(text)
    ids = [j for j in ids if j != tokenizer.unk_id]
    return ids

def encode(tokenizer, i, target_span_len):

    task = task_ids[i['mode']]

    ids = []
    info = [task]

    if 'control' in i:
        control = get_control(i['control'], tokenizer, task)
    else:
        control = []

    ids += control
    info.append(len(control))
    
    assert len(i['source']) <= 2
    src = i['source'][0]

    src_ids = convert_to_ids(tokenizer, src)
    src_ids = [tokenizer.bos_id] + src_ids
    if task != 0:
         src_ids += [tokenizer.eos_id]
    ids += src_ids
    info.append(len(src_ids))

    tgt_ids = [0] * target_span_len
    # 这里的bos、eos可能只用于占位
    tgt_ids = [tokenizer.bos_id] + tgt_ids + [tokenizer.eos_id]
    ids += tgt_ids
    if task == 0:
        # 续写（只预测single span)
        if len(i['source']) > 1 and len(i['source'][1]) > 0:
            # 生成中间，可以看见eos
            end_ids = convert_to_ids(tokenizer, i['source'][1])
            end_ids += [tokenizer.eos_id]
            ids += end_ids
            info.extend([len(tgt_ids), len(end_ids)])
        else:
            # 生成结尾
            # 可以自由生成
            info.extend([len(tgt_ids), 0])
            # 可以指定长度，给定eos
            # info.extend([len(tgt_ids)-1, 1])
    elif task == 1 or task == 2:
        # 压缩、扩写
        # 可以自由生成
        info.extend([len(tgt_ids), 0])
        # 可以提供eos
        # info.extend([len(tgt_ids)-1, 1])
    else:
        # 改写和改错不控制长度，eos自己生成
        info.extend([len(tgt_ids), 0])
        
    info = info[:1] + np.cumsum(info[1:]).tolist()

    assert len(ids) == info[-1]
    assert len(info) % 2 == 1 # task, control, src, tgt, src, tgt, ...., end
    return ids, info


def encode_ppl(tokenizer, i):
    task = task_ids[i['mode']]

    ids = []
    info = [task]

    if 'control' in i:
        control = get_control(i['control'], tokenizer, task)
    else:
        control = []

    ids += control
    info.append(len(control))
    
    assert len(i['source']) <= 2
    src = i['source'][0]

    src_ids = convert_to_ids(tokenizer, src)
    src_ids = [tokenizer.bos_id] + src_ids
    if task != 0:
         src_ids += [tokenizer.eos_id]
    ids += src_ids
    info.append(len(src_ids))

    tgt = i['target']
    if isinstance(tgt, list):
        assert len(tgt) == 1
        tgt = tgt[0]
    else:
        assert isinstance(tgt, str)

    tgt_ids = convert_to_ids(tokenizer, tgt)
    if task == 0:
        # 续写（只预测single span)
        if len(i['source']) > 1 and len(i['source'][1]) > 0:
            # 生成中间，可以看见eos
            end_ids = convert_to_ids(tokenizer, i['source'][1])
            end_ids += [tokenizer.eos_id]
            ids += tgt_ids
            ids += end_ids
            info.extend([len(tgt_ids), len(end_ids)])
        else:
            # 生成结尾
            # 可以自由生成
            ids += tgt_ids
            ids += [tokenizer.eos_id]
            info.extend([len(tgt_ids) + 1, 0])
            # 可以指定长度，给定eos
            # info.extend([len(tgt_ids), 1])
    elif task == 1 or task == 2:
        # 压缩、扩写
        # 可以自由生成
        tgt_ids = [tokenizer.bos_id] + tgt_ids + [tokenizer.eos_id]
        ids += tgt_ids
        info.extend([len(tgt_ids), 0])
        # 可以提供eos
        # info.extend([len(tgt_ids)-1, 1])
    else:
        tgt_ids = [tokenizer.bos_id] + tgt_ids + [tokenizer.eos_id]
        ids += tgt_ids
        # 改写和改错不控制长度，eos自己生成
        info.extend([len(tgt_ids), 0])
        
    info = info[:1] + np.cumsum(info[1:]).tolist()

    assert len(ids) == info[-1]
    assert len(info) % 2 == 1 # task, control, src, tgt, src, tgt, ...., end
    return ids, info


def generate_no_beam_cpm3(model, tokenizer, instance, target_span_len,
                     temperature = .9, top_k = 0, top_p = 0.9,
                     no_repeat_ngram_size = 0, repetition_penalty = 1, random_sample=False, min_len=None):

    ids, info = encode(tokenizer, instance, target_span_len)

    prompt_length = 64
    input_tokens, input_length, context_input, position_input, segment_input, span_input, _ = make_input_cpm3(ids, info, prompt_length)

    # torch.set_printoptions(profile="full")
    # print("input_tokens")
    # print(input_tokens)
    # print("input_length")
    # print(input_length)
    # print("context_input")
    # print(context_input)
    # print("position_input")
    # print(position_input)
    # print("segment_input")
    # print(segment_input)
    # print("span_input")
    # print(span_input)

    input_tokens = input_tokens.int().cuda()
    input_length = input_length.int().cuda()
    context_input = context_input.bool().cuda()
    position_input = position_input.float().cuda()
    segment_input = segment_input.int().cuda()
    span_input = span_input.int().cuda()

    lef = info[2] + prompt_length
    rig = info[3] + prompt_length

    # bmp.print_rank(lef, rig)
    with torch.inference_mode():
        for i in range(lef - 1, rig - 1):
            logits, _ = model(input_tokens, input_length, context_input, position_input, segment_input, span_input)
            # print("current {}".format(i))
            # torch.set_printoptions(profile="full")
            # print(logits[:, i, :])
            logits = logits[:, i, :] / temperature
            logits = postprocess_next_token_scores(
                tokenizer=tokenizer,
                scores=logits,
                input_ids=input_tokens,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=[[0]],
                repetition_penalty=repetition_penalty,
                batch_size=1,
                num_beams=1,
                start_idx=lef,
                end_idx=i,
                window_size=None,
                min_len=min_len
            )
            logits = top_k_logits(logits, top_k=top_k, top_p=top_p)

            if random_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = logits.argmax(dim=-1)

            input_tokens[0][i + 1] = next_token

            # early stop, Note: not supports multi-GPUs
            if next_token == tokenizer.eos_id:
                break

        for idx, id in enumerate(input_tokens[0].cpu().numpy()):
            token = tokenizer.decode([id])
            if idx < prompt_length:
                continue
            
            if id == tokenizer.pad_id:
                break

            yield token
            

def generate_contrastive_search_cpm3(model, tokenizer, instance, target_span_len,
                     top_k = 5, alpha=0.7,
                     no_repeat_ngram_size = 0, repetition_penalty = 1, random_sample=False, min_len=None):

    ids, info = encode(tokenizer, instance, target_span_len)

    prompt_length = 64
    input_tokens, input_length, context_input, position_input, segment_input, span_input, _ = make_input_cpm3(ids, info, prompt_length)

    input_tokens = input_tokens.int().cuda()
    input_length = input_length.int().cuda()
    context_input = context_input.bool().cuda()
    position_input = position_input.float().cuda()
    segment_input = segment_input.int().cuda()
    span_input = span_input.int().cuda()

    lef = info[2] + prompt_length
    rig = info[3] + prompt_length

    # bmp.print_rank(lef, rig)
    with torch.inference_mode():
        for i in range(lef - 1, rig - 1):
            logits, hidden_states = model(input_tokens, input_length, context_input, position_input, segment_input, span_input)
            # print("current {}".format(i))
            # torch.set_printoptions(profile="full")
            # print(logits[:, i, :])
            logits = logits[:, i, :]
            logits = postprocess_next_token_scores(
                tokenizer=tokenizer,
                scores=logits,
                input_ids=input_tokens,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=[[0]],
                repetition_penalty=repetition_penalty,
                batch_size=1,
                num_beams=1,
                start_idx=lef,
                end_idx=i,
                window_size=None,
                min_len=min_len
            )
            
            cur_hiddens = hidden_states[:, lef:i+1, :].float()
            probs = F.softmax(logits, dim=-1)
            cand_probs, cand_ids = torch.topk(probs, top_k, dim=-1, largest=True, sorted=True)
            
            if cur_hiddens.size(1) == 0:
                best_id = cand_ids[0, 0]
            else:
                # slow implementation (can batchify them to optimize)
                best_id = None
                best_score = None
                cur_hiddens_norm = cur_hiddens / cur_hiddens.norm(dim=2, keepdim=True)
                for k_i in range(top_k):
                    input_tokens[0][i + 1] = cand_ids[0][k_i]
                    _, hidden_states = model(input_tokens, input_length, context_input, position_input, segment_input, span_input)
                    next_hidden = hidden_states[:, i + 1, :].unsqueeze(1).float()
                    next_hidden_norm = next_hidden/ next_hidden.norm(dim=2, keepdim=True)
                    cosine_matrix = torch.matmul(cur_hiddens_norm, next_hidden_norm.transpose(1, 2)).squeeze(-1)
                    max_score, _ = torch.max(cosine_matrix, dim=-1)
                    
                    score = (1 - alpha) * cand_probs[0][k_i] - alpha * max_score

                    # print(tokenizer.decode([cand_ids[0, k_i].item()]), score, cand_probs[0][k_i], max_score)

                    if best_score is None or score > best_score:
                        best_score = score
                        best_id = cand_ids[0][k_i]
            
            # print(tokenizer.decode(input_tokens[0].cpu().tolist()[64:i+1]), " + ", str(tokenizer.decode([best_id.item()])))
            # print("=================================")

            input_tokens[0][i + 1] = best_id

        for idx, id in enumerate(input_tokens[0].cpu().numpy()):
            token = tokenizer.decode([id])
            if idx < prompt_length:
                continue
            
            if id == tokenizer.pad_id:
                break

            yield token


def generate_beam(model, tokenizer, instance, target_span_len, beam_size = 3,
                     temperature = .9, top_k = 0, top_p = 0.9,
                     no_repeat_ngram_size = 0, repetition_penalty = 1, random_sample=False, min_len=None):
    
    vocab_size = tokenizer.vocab_size

    ids, info = encode(tokenizer, instance, target_span_len)

    prompt_length = 64
    input_tokens, input_length, context_input, position_input, segment_input, span_input, _ = make_input_cpm3(ids, info, prompt_length)

    # (batch, max_length) 
    max_length = input_tokens.size(-1)
    batch_size = input_tokens.size(0)

    # (batch, beam_size, max_length)    
    input_tokens = input_tokens.unsqueeze(1).expand(batch_size, beam_size, max_length)
    input_length = input_length.unsqueeze(1).expand(batch_size, beam_size)
    span_input = span_input.unsqueeze(1).expand(batch_size, beam_size, max_length)
    context_input = context_input.unsqueeze(1).expand(batch_size, beam_size, max_length)
    position_input = position_input.unsqueeze(1).expand(batch_size, beam_size, max_length)
    segment_input = segment_input.unsqueeze(1).expand(batch_size, beam_size, max_length)
    # (batch * beam_size, max_length)    
    input_tokens = input_tokens.contiguous().view(batch_size * beam_size, max_length)
    input_length = input_length.contiguous().view(batch_size * beam_size,)
    span_input = span_input.contiguous().view(batch_size * beam_size, max_length)
    context_input = context_input.contiguous().view(batch_size * beam_size, max_length)
    position_input = position_input.contiguous().view(batch_size * beam_size, max_length)
    segment_input = segment_input.contiguous().view(batch_size * beam_size, max_length)

    input_tokens = input_tokens.int().cuda()
    input_length = input_length.int().cuda()
    context_input = context_input.bool().cuda()
    position_input = position_input.float().cuda()
    segment_input = segment_input.int().cuda()
    span_input = span_input.int().cuda()

    done = [False for _ in range(batch_size)]
    # (batch_size * beam_size, 0)
    
    beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=input_tokens.device)
    beam_scores[:, 1:] = -1e9 # 确保第一次只在一个vocab大小里选取
    beam_scores = beam_scores.view(-1)

    # current position
    cur_len = 0

    lef = info[2] + prompt_length
    rig = info[3] + prompt_length

    span_length = rig - lef

    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(beam_size, span_length, length_penalty=1 , early_stopping=False, tokenizer=tokenizer)
        for _ in range(batch_size)
    ]


    bmp.print_rank(lef, rig)
    with torch.inference_mode():
        for i in range(lef - 1, rig - 1):
            logits, _ = model(input_tokens, input_length, context_input, position_input, segment_input, span_input)

            # skip all steps when we are done with each sentence
            if all(done):
                break # Note: break not supports multi-GPUs
    
            # (batch * beam, seqlen, model_dim)
            logits = logits[:, i, :]
            logits = postprocess_next_token_scores(
                tokenizer=tokenizer,
                scores=logits,
                input_ids=input_tokens,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=[[0]],
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=beam_size,
                start_idx=lef,
                end_idx=i,
                window_size=None,
                min_len=min_len
            )
            scores = F.log_softmax(logits, dim=-1)

            if random_sample:
                # TODO: need to check this part
                assert temperature != 0, "temperature should not be zero!"
                scores = scores - math.log(temperature)
                _scores = scores + beam_scores[:, None].expand_as(scores)
                             
                _scores = top_k_logits(_scores, top_k=top_k, top_p=top_p)
                _scores = _scores.contiguous().view(batch_size, beam_size * vocab_size)
                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                probs = F.softmax(_scores, dim=-1)
                next_words = torch.multinomial(probs, num_samples=2 * beam_size)  # (batch_size, beam_size * 2)
                # Compute next scores
                next_scores = torch.gather(_scores, -1, next_words)  # (batch_size, beam_size * 2)
                # sort the sampled vector to make sure that the first beam_size samples are the best
                next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
                next_words = torch.gather(next_words, -1, next_scores_indices)  # (batch_size, beam_size * 2)            
            else:
                next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * beam_size, vocab_size)

                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                next_scores = next_scores.view(
                    batch_size, beam_size * vocab_size
                )  # (batch_size, beam_size * vocab_size)

                next_scores, next_words = torch.topk(next_scores, 2 * beam_size, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_words.size() == (batch_size, 2 * beam_size)
            # next batch beam content
            next_batch_beam = []

            for sent_id in range(batch_size):

                 # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item(), cur_len)
                if done[sent_id]:
                    next_batch_beam.extend([(0, tokenizer.pad_id, 0)] * beam_size)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size

                    # end of sentence, or next word
                    if word_id == tokenizer.eos_id or cur_len + 1 == span_length:
                        if cur_len > 0:
                            generated_hyps[sent_id].add(input_tokens[sent_id * beam_size + beam_id, lef:lef+cur_len].clone(), value.item())
                    # elif cur_len + 1 == span_length:
                    #     # 没有正常结束，指定为很低的分数
                    #     generated_hyps[sent_id].add(input_tokens[sent_id * beam_size + beam_id, lef:lef+cur_len].clone(), -50000)
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == span_length else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, tokenizer.pad_id, 0)] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input_tokens.new([x[1] for x in next_batch_beam])
            beam_idx = input_length.new([x[2] for x in next_batch_beam]).long()

            # re-order batch and internal states
            input_tokens = input_tokens[beam_idx, :]
            input_tokens[:, lef + cur_len] = beam_words

            # update current length
            cur_len = cur_len + 1

        # select the best hypotheses
        tgt_len = input_length.new(batch_size)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # because batch_size = 1        
        for id in best[0].cpu().numpy():
            token = tokenizer.decode([id])

            yield token


def generate(model, tokenizer, instance, target_span_len, beam,
                     temperature = .9, top_k = 0, top_p = 0.9,
                     no_repeat_ngram_size = 0, repetition_penalty = 1,
                     random_sample=False, min_len=None):
    if beam == 1:
        generation_str = generate_no_beam_cpm3(model, tokenizer, instance, target_span_len,
                                        temperature, top_k, top_p,
                                        no_repeat_ngram_size, repetition_penalty, random_sample, min_len)
        # generation_str = generate_contrastive_search_cpm3(model, tokenizer, instance, target_span_len,
        #                                  5, 0.7, 
        #                                 no_repeat_ngram_size, repetition_penalty, random_sample, min_len)
    else:
        generation_str = generate_beam(model, tokenizer, instance, target_span_len, beam,
                                    temperature, top_k, top_p,
                                    no_repeat_ngram_size, repetition_penalty, random_sample, min_len)

    return generation_str


def get_ppl(model, tokenizer, instance):
    ids, info = encode_ppl(tokenizer, instance)

    prompt_length = 64
    input_tokens, input_length, context_input, position_input, segment_input, span_input, tgt_input = make_input_cpm3(ids, info, prompt_length)

    input_tokens = input_tokens.int().cuda()
    input_length = input_length.int().cuda()
    context_input = context_input.bool().cuda()
    position_input = position_input.float().cuda()
    segment_input = segment_input.int().cuda()
    span_input = span_input.int().cuda()
    target_input = tgt_input.long().cuda()
    
    with torch.inference_mode():
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
        logits = model(input_tokens, input_length, context_input, position_input, segment_input, span_input)
        loss = loss_func(logits.view(-1, logits.size(-1)), target_input.view(-1))
    
    return math.exp(loss.item())
