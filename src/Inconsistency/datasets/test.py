import torch
from transformers import pipeline

classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    return_all_scores=True,
    dtype=torch.float16,
    device=0
)

result = classifier("I love using Hugging Face Transformers!")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]