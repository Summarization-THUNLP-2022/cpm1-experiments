 # 实验记录

## 简介

- 使用LCSTS数据集进行实验

## 对代码的修改

- 添加了cpm1_dataset.py，LCSTS_Dataset类用于加载LCSTS数据集
- 修改finetune_cpm1.py中的finetune代码适用于LCSTS等摘要生成任务
- 修改loss为字符平均loss

## 结果

- 跑了一天半之后，发现未添加结束符。。
