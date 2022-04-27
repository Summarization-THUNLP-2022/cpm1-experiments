import torch
import torch.nn.functional as F
import numpy as np
import bmtrain as bmt


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


def generate_beam(model, tokenizer, lef_sentence, spans, beam_size = 16, beam_group= 4, diverse_penalty=0.5):
    assert beam_size % beam_group == 0
    beam_size_group = beam_size // beam_group

    vocab_size = tokenizer.vocab_size

    lef_tokens = tokenizer.encode(lef_sentence)
    lef_tokens = [1] + lef_tokens

    input_tokens, input_length, input_span, context = make_input(lef_tokens, spans)

    max_length = input_tokens.size(-1)
    batch_size = input_tokens.size(0)

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
        DiverseBeamHypotheses(beam_size, beam_group, spans, length_penalty=1, early_stopping=False, tokenizer=tokenizer)
        for _ in range(batch_size)
    ]

    lef = len(lef_tokens)
    rig = len(lef_tokens) + spans

    # bmp.print_rank(lef, rig)
    with torch.inference_mode():
        for i in range(lef-1, rig-1):
            logits = model(input_tokens, input_length, context, input_span)

            logits = logits[:, i, :]
            logits[:, [0, 1, 2, 3] + [5] + [x for x in range(8, 20)]] = -float("inf")
            scores = F.log_softmax(logits, dim=-1)
            
            next_scores_all = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * beam_size, vocab_size)

            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            next_scores_all = next_scores_all.view(
                batch_size, beam_group, beam_size_group, vocab_size
            )

            next_batch_beam = []
            for g in range(beam_group):
                for sent_id in range(batch_size):
                    idx_penalty = set()
                    for beam in next_batch_beam[: g * beam_size_group]:
                        idx_penalty.add(int(beam[1]))
                    idx_penalty = list(idx_penalty)
                    next_scores_all[sent_id,g,:,idx_penalty] -= diverse_penalty
                next_scores = next_scores_all.view(batch_size, beam_group, beam_size_group * vocab_size)

                next_scores, next_words = torch.topk(next_scores, 2 * beam_size_group, dim=2, largest=True, sorted=True)

                # next batch beam content

                for sent_id in range(batch_size):

                    # if we are done with this sentence
                    done[sent_id][g] = done[sent_id][g] or generated_hyps[sent_id].is_done(g, next_scores[sent_id][g].max().item(), cur_len)
                    if done[sent_id][g]:
                        next_batch_beam.extend([(0, tokenizer.pad_id, 0)] * beam_size_group)  # pad the batch
                        continue

                    # next sentence beam content
                    next_sent_beam = []

                    # next words for this sentence
                    for idx, value in zip(next_words[sent_id][g], next_scores[sent_id][g]):

                        # get beam and word IDs
                        beam_id = idx // vocab_size
                        word_id = idx % vocab_size

                        # end of sentence, or next word
                        if word_id == tokenizer.eod_id or cur_len + 1 == spans:
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
                    assert len(next_sent_beam) == 0 if cur_len + 1 == spans else beam_size
                    if len(next_sent_beam) == 0:
                        next_sent_beam = [(0, tokenizer.pad_id, 0)] * beam_size_group  # pad the batch
                    next_batch_beam.extend(next_sent_beam)

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


def generate(model, tokenizer, instance, target_span_len, beam, beam_group, diverse_penalty):
    if beam == 1:
        pass
    else:
        generation_str = generate_beam(model, tokenizer, instance, target_span_len, beam, beam_group, diverse_penalty)

    return generation_str
