# Use a pipeline as a high-level helper
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)