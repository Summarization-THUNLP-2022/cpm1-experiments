# %%
import torch

# %%
model1 = torch.load("/data2/private/zhaoxinhao/ModelCenter/cpm1-small/pytorch_model.pt")
model2 = torch.load("/home/zhaoxinhao/data2/cpm1/experiments/20220505_1_20220424_1_cpm1small_infer/results/finetune-cpm1-ckpt-4-0.pt")
for key, value in model1.items():
	if model2[key].shape != value.shape:
		print(key)

# %%
from model_center.model import CPM1
# %%
model = CPM1.from_pretrained("/data2/private/zhaoxinhao/ModelCenter/cpm1-small")
# %%
