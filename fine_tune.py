# |--File name:   --|fine_tune
# |--File path:   --|
# |--Create Date: --|2025/9/2
# |--Programmer:  --|PY.Wang
# |--Description: --|Description
# |--version:     --|

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


# Loading the Model and Tokenizer
model_path = "saved_models/Qwen3-0.6B"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype="auto",
    device_map=device
)

# Model Configuration:
model.config.use_cache = False
model.config.pretraining_tp = 1

# Loading and Processing the Dataset
def formatting_prompts_func(series: pd.Series):
    question = series.loc["Question"]
    response = series.loc["Response"]
    # Append the EOS token to the response if it's not already there
    if not response.endswith(tokenizer.eos_token):
        response += tokenizer.eos_token
    text = f"""<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{response}"""
    return text

dataset = pd.read_excel('dataset/chuunibyo.xlsx')
dataset['text'] = dataset.apply(
    formatting_prompts_func,
    axis=1
)
dataset.to_json('dataset/data.json', orient='records', force_ascii=False, indent=4)
train_data = load_dataset('json', data_files='dataset/data.json', split='train')

# The new STF trainer does not accept the tokenizer, so we will convert the tokenizer
# into a data collator using the simple transformer function.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# Setting up the Model
# LoRA config
peft_config = LoraConfig(
    lora_alpha=16,                           # Scaling factor for LoRA
    lora_dropout=0.05,                       # Add slight dropout for regularization
    r=64,                                    # Rank of the LoRA update matrices
    bias="none",                             # No bias reparameterization
    task_type="CAUSAL_LM",                   # Task type: Causal Language Modeling
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Target modules for LoRA
)

# We will now configure and initialize the SFTTrainer (Supervised Fine-Tuning Trainer),
# a high-level abstraction provided by Hugging Face's transformers and trl libraries.
# The SFTTrainer simplifies the fine-tuning process by integrating key components—such
# as the dataset, model, data collator, training arguments, and LoRA configuration—into a single, streamlined workflow


# Training Arguments
training_arguments = TrainingArguments(
    output_dir="checkpoint_output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=10,
    logging_steps=0.2,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to="none"
)

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_data,
    peft_config=peft_config,
    data_collator=data_collator,
)

# Model Training
torch.cuda.empty_cache()
trainer.train()

# save model
merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained("saved_models/Qwen3-0.6B-chuunibyou")
tokenizer.save_pretrained("saved_models/Qwen3-0.6B-chuunibyou")
