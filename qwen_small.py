#|--Create Date: --|2025-09-02
#|--Programmer:  --|Wang Pengyu
#|--Description: --|
#|--version:     --|

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import pandas as pd
from chat import text_generate, text_generate_no_stream

# model_path = "saved_models/Qwen3-0.6B"
model_path = "saved_models/Qwen3-0.6B-chuunibyou"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype="auto",
    device_map=device
)

if __name__ == "__main__":
    answer = text_generate_no_stream(
        '请你做一下自我介绍',
        model,
        tokenizer,
    )
