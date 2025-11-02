# train.py (Based on previous Step 4)
import os
import torch
import argparse
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from huggingface_hub import HfFolder

# --- Configuration Constants ---
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" 
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "output_weights"
DATASET_PATH = "your_chat_data.jsonl" # <<--- ADJUST THIS TO YOUR FILE/URL!

# --- Training Data Formatting ---
# Use the Llama 3 format for instruction tuning
def formatting_prompts_func(examples):
    texts = []
    # Assumes 'conversation' key in your JSONL holds the chat list, 
    # e.g., [{"role": "user", "content": "Hi"}, ...]
    for conversation in examples.get('conversation', []):
        # Apply Llama 3 chat template from Unsloth's tokenizer
        text = FastLanguageModel.get_llama_3_prompt(conversation, tokenizer)
        texts.append(text)
    return { "text" : texts }

# --- Main Training Function ---
def train(args):
    # 1. Load Model (Quantized 4-bit QLoRA)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None, # Auto-detects bfloat16 for A100
        load_in_4bit = True, # QLoRA 4-bit quantization
        device_map = "auto",
        token = os.environ.get("HF_TOKEN") # Fetches secret
    )

    # 2. Add LoRA Adapters (The only part that is fully trained)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank of the LoRA matrices.
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        max_seq_length = MAX_SEQ_LENGTH,
    )

    # 3. Load and Format Data
    # Assuming 'json' file type, adjust if needed (e.g., 'csv')
    try:
        dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    except Exception as e:
        print(f"ERROR: Failed to load dataset from {DATASET_PATH}. Please check file name/path.")
        raise e
        
    dataset = dataset.map(
        formatting_prompts_func, 
        batched = True,
        # Only keep the required 'text' column for training
        remove_columns = list(dataset.features.keys()), 
    )

    # 4. Define Training Arguments
    training_args = TrainingArguments(
        per_device_train_batch_size = 4, 
        gradient_accumulation_steps = 2, # Effective batch size = 8
        num_train_epochs = 3, 
        learning_rate = 2e-4,
        bf16 = True, # Use bfloat16 on A100
        logging_steps = 1,
        output_dir = OUTPUT_DIR,
        optim = "adamw_8bit", # Memory-efficient 8-bit optimizer
        seed = 42,
        save_strategy = "epoch",
    )

    # 5. Initialize and Run Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        args = training_args,
    )
    
    print("Starting Llama 3 8B QLoRA Fine-Tuning...")
    trainer.train()

    # 6. Save Model
    trainer.model.save_pretrained(OUTPUT_DIR) 
    tokenizer.save_pretrained(OUTPUT_DIR) # Save tokenizer in output dir too
    print(f"Training complete. Weights saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    train(args)