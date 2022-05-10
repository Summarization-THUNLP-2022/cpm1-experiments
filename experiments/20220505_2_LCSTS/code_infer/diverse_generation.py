import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


class DiverseBeamHypotheses:
    def __init__(self, n_hyp, n_group, max_len, length_penalty, early_stopping, tokenizer=None):
        self.max_len = max_len
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.n_group = n_group
        self.hyp = [[] for _ in range(n_group)]
        self.worst_score = [1e9] * n_group
        self.tokenizer = tokenizer
        assert self.n_hyp % self.n_group == 0
        self.group_hyp = self.n_hyp / self.n_group


    def add(self, group, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty

        if len(self.hyp[group]) < self.group_hyp or score > self.worst_score[group]:
            self.hyp[group].append((score, hyp))
            if len(self.hyp[group]) > self.group_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp[group])])
                del self.hyp[group][sorted_scores[0][1]]
                self.worst_score[group] = sorted_scores[1][0]
            else:
                self.worst_score[group] = min(score, self.worst_score[group])
        
    def is_done(self, group, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self.hyp[group]) < self.group_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score[group] >= best_sum_logprobs / cur_len ** self.length_penalty


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
    return banned_tokens


# min_length_constriant
def min_length_constraint(logits, cur_len, min_len, tokenizer):
    # This enforcing a min-length by setting EOS probability to 0.
    if cur_len < min_len:
        logits[:, tokenizer.eod_id] = -float("inf")


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

    # 允许生成eos
    scores[:, [0, 1, 2, 3] + [x for x in range(5, 8)]] = -float("inf")

    if start_idx is not None and end_idx is not None and end_idx >= start_idx and min_len is not None:
        min_length_constraint(scores, end_idx - start_idx + 1, min_len, tokenizer)

    return scores


def round_up(x, d):
    return (x + d - 1) // d * d


def make_input(lef_tokens, spans):
    input = lef_tokens + [0 for i in range(spans)]
    length = len(input)

    rounded_length = round_up(length, 4)

    input_tokens = torch.zeros(1, rounded_length, dtype=torch.int32)
    input_span = torch.zeros(1, rounded_length, dtype=torch.int32)
    
    context = np.arange((rounded_length))
    context = (context < len(lef_tokens)) | (context >= len(lef_tokens) + spans)
    context = torch.from_numpy(context).view(1, -1).bool()

    input_length = torch.zeros(1, dtype=torch.int32)
    input_tokens[0, :length] = torch.tensor(input).int()
    input_length[0] = length

    return input_tokens.cuda(), input_length.cuda(), input_span.cuda(), context.cuda()


def generate_beam(model, tokenizer, input_dict, beam_size = 16, beam_group= 4, diverse_penalty=0.5, no_repeat_ngram_size = 0, repetition_penalty = 1, min_len=None):
    assert beam_size % beam_group == 0
    beam_size_group = beam_size // beam_group

    vocab_size = tokenizer.vocab_size

    input_tokens = input_dict['input_tokens'].cuda()
    input_span = input_dict['input_span'].cuda()
    context = input_dict['context'].cuda()
    source_length = input_dict['source_length']

    batch_size = input_tokens.size(0)
    max_length = input_tokens.size(-1)
    span_length = max_length - source_length

    input_length = max_length * torch.ones([batch_size], dtype=torch.int32).cuda()


    input_tokens = input_tokens.unsqueeze(1).expand(batch_size, beam_size, max_length)
    input_length = input_length.unsqueeze(1).expand(batch_size, beam_size)
    input_span = input_span.unsqueeze(1).expand(batch_size, beam_size, max_length)
    context = context.unsqueeze(1).expand(batch_size, beam_size, max_length)

    input_tokens = input_tokens.contiguous().view(batch_size * beam_size, max_length)
    input_length = input_length.contiguous().view(batch_size * beam_size,)
    input_span = input_span.contiguous().view(batch_size * beam_size, max_length)
    context = context.contiguous().view(batch_size * beam_size, max_length)

    done = [[False for _ in range(beam_group)] for _ in range(batch_size)]
    
    beam_scores = torch.zeros((batch_size, beam_group, beam_size_group), dtype=torch.float, device=input_tokens.device)
    beam_scores[:, :, 1:] = -1e9
    beam_scores = beam_scores.view(-1)

    cur_len = 0
    
    generated_hyps = [
        DiverseBeamHypotheses(beam_size, beam_group, span_length, length_penalty=1, early_stopping=False, tokenizer=tokenizer)
        for _ in range(batch_size)
    ]

    lef = source_length
    rig = max_length

    with torch.inference_mode():
        past_key_values = None
        for i in range(lef-1, rig-1):
            if sum([sum(d) for d in done]) == sum([len(d) for d in done]):
                break
            
            if i == lef-1:
                logits, past_key_values = model(input_tokens[:, :i+1], input_length, context[:, :i+1], input_span[:, :i+1], past_key_values)
                logits = logits[:, -1, :]
            else:
                logits, past_key_values = model(input_tokens[:, i:i+1], input_length, context[:, :i+1], input_span[:, :i+1], past_key_values)
                logits = logits[:, -1, :]
            
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
            
            next_scores_all = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * beam_size, vocab_size)

            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            next_scores_all = next_scores_all.view(
                batch_size, beam_group, beam_size_group, vocab_size
            )

            next_batch_beam_group = [[] for _ in range(batch_size)]

            for g in range(beam_group):
                for sent_id in range(batch_size):
                    for beam in next_batch_beam_group[sent_id][: g * beam_size_group]:
                        next_scores_all[sent_id, g, :, int(beam[1])] -= diverse_penalty
                next_scores = next_scores_all.view(batch_size, beam_group, beam_size_group * vocab_size)

                next_scores, next_words = torch.topk(next_scores, 2 * beam_size_group, dim=2, largest=True, sorted=True)

                # next batch beam content

                for sent_id in range(batch_size):

                    # if we are done with this sentence
                    done[sent_id][g] = done[sent_id][g] or generated_hyps[sent_id].is_done(g, next_scores[sent_id][g].max().item(), cur_len)
                    if done[sent_id][g]:
                        next_batch_beam_group[sent_id].extend([(0, tokenizer.pad_id, 0)] * beam_size_group)  # pad the batch
                        continue

                    # next sentence beam content
                    next_sent_beam = []

                    # next words for this sentence
                    for idx, value in zip(next_words[sent_id][g], next_scores[sent_id][g]):

                        # get beam and word IDs
                        beam_id = idx // vocab_size
                        word_id = idx % vocab_size

                        # end of sentence, or next word
                        if word_id == tokenizer.eod_id or cur_len + 1 == span_length:
                            if cur_len > 0:
                                generated_hyps[sent_id].add(g, input_tokens[sent_id * beam_size + g * beam_size_group + beam_id, lef:lef+cur_len].clone(), value.item())
                        # elif cur_len + 1 == span_length:
                        #     # 没有正常结束，指定为很低的分数
                        #     generated_hyps[sent_id].add(input_tokens[sent_id * beam_size + beam_id, lef:lef+cur_len].clone(), -50000)
                        else:
                            next_sent_beam.append((value, word_id, sent_id * beam_size + g * beam_size_group + beam_id))

                        # the beam for next step is full
                        if len(next_sent_beam) == beam_size_group:
                            break

                    # update next beam content
                    assert len(next_sent_beam) == 0 if cur_len + 1 == span_length else beam_size
                    if len(next_sent_beam) == 0:
                        next_sent_beam = [(0, tokenizer.pad_id, 0)] * beam_size_group  # pad the batch
                    next_batch_beam_group[sent_id].extend(next_sent_beam)

            next_batch_beam = [sent_beam for sent_beam_group in next_batch_beam_group for sent_beam in sent_beam_group]

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input_tokens.new([x[1] for x in next_batch_beam])
            beam_idx = input_length.new([x[2] for x in next_batch_beam]).long()

            # re-order batch and internal states
            input_tokens = input_tokens[beam_idx, :]
            input_tokens[:, lef + cur_len] = beam_words

            for key_value_layer in past_key_values:
                key_value_layer[0] = key_value_layer[0][beam_idx]
                key_value_layer[1] = key_value_layer[1][beam_idx]

            # update current length
            cur_len = cur_len + 1


        result = []

        for i, hypotheses in enumerate(generated_hyps):
            for group in hypotheses.hyp:
                for hyp in group:
                    hyp_sent = ""
                    for id in hyp[1]:
                        token = tokenizer.decode([int(id)])
                        if token == '<eod>':
                            break
                        hyp_sent += token
                    result.append(hyp_sent)
        return result


def diverse_beam_search_generate(model, tokenizer, input_dict, beam_size = 16, beam_group= 4, diverse_penalty=0.5, no_repeat_ngram_size = 0, repetition_penalty = 1, min_len=None):
    if beam_size == 1:
        pass
    else:
        generation_str = generate_beam(model, tokenizer, input_dict, beam_size=beam_size, beam_group=beam_group, diverse_penalty=diverse_penalty, no_repeat_ngram_size=no_repeat_ngram_size, repetition_penalty=repetition_penalty, min_len=None)

    return generation_str
