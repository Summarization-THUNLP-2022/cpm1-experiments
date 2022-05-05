# BRIO输入输出流程

## 文件格式（Dataset输入）

json list

```json
{"lef_tokens": [...], "rig_tokens": [...], "cand_rig_tokens": [[...], [...], [...]]}
```
candidates按生成质量排序

## Dataset输出

- input_tokens: [batch_size, candidate_num, seq_length]
- input_length: [batch_size, candidate_num]
- context: [batch_size, candidate_num, seq_length]
- input_span: [batch_size, candidate_num, seq_length]
- target: [batch_size, candidate_num, seq_length]
- target_length: [batch_size, candidate_num]

## 模型输入

- input_tokens: [batch_size * candidate_num, seq_length]
- input_length: [batch_size * candidate_num]
- context: [batch_size * candidate_num, seq_length]
- input_span: [batch_size * candidate_num, seq_length]

## 模型输出

- logits: [batch_size * candidate_num, seq_length, vocab_size]

## 计算loss
