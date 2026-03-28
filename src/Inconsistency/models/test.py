from transformers import AutoModel, AutoTokenizer
import torch

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

model = AutoModel.from_pretrained("FacebookAI/roberta-base").to("cuda")
tokenizer=AutoTokenizer.from_pretrained("FacebookAI/roberta-base")


inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    outputs=model(**inputs)

print("="*30)
print(outputs.last_hidden_state.shape)
breakpoint()