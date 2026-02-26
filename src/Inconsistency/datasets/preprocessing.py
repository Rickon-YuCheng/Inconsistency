# 提取正負中性標籤
import pdb
import torch
from transformers import pipeline


def DISTELBERT():
    classifier = pipeline(
        task="text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        dtype=torch.float16,
        device=0)
    while true:
        with open()
            result = classifier("I love using Hugging Face Transformers!")
        print(result)
        print(f"polarity: {result[0]['label']}")


# Output: [{'label': 'POSITIVE', 'score': 0.9998}]

if __name__ == '__main__':
    DISTELBERT()
    WAV2VEC2()