# predict.py
import os
import torch
import sys
from unsloth import FastLanguageModel
from cog import BasePredictor, Input, Path

# --- Configuration Constants ---
# MUST MATCH the model name used in train.py
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" 
MAX_SEQ_LENGTH = 2048 # Max context length for inference
OUTPUT_ADAPTERS = "output_weights" # Directory where train.py saves adapters

# Add the output directory to the path so modules can be imported
sys.path.append(OUTPUT_ADAPTERS)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model and tokenizer into memory for prediction"""
        print("Starting model setup: Loading Llama 3 base and merging LoRA adapters...")
        
        # 1. Load the base model (unsloth optimized)
        # We load in 4-bit for memory efficiency even during inference
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = MODEL_NAME,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = None, # Auto-detects bfloat16 for A100/H100
            load_in_4bit = True, 
            # Use your Hugging Face token (set as a secret on Replicate)
            token = os.environ.get("HF_TOKEN") 
        )

        # 2. Load the fine-tuned LoRA adapters
        try:
            self.model = FastLanguageModel.get_peft_model(self.model)
            self.model.load_adapter(OUTPUT_ADAPTERS)
        except Exception as e:
             # This block handles the case where setup runs before training is complete
            print(f"Warning: Could not load LoRA adapters from {OUTPUT_ADAPTERS}. Error: {e}")
            print("Proceeding with base model for testing.")

        # 3. Merge LoRA weights into the base model for faster inference
        # This is a critical step for maximizing chat performance
        self.model.base_model.merge_and_unload()
        
        # Ensure model is ready for generation
        self.model.to(dtype=torch.bfloat16).cuda()
        self.tokenizer.pad_token = self.tokenizer.eos_token 

        print("Model loading and merging complete. Ready for chat.")

    def predict(
        self,
        prompt: str = Input(description="The user's input/question for the chat model.",),
        system_prompt: str = Input(
            description="The system instruction to guide the model's style, role, and IP.",
            default="You are a helpful and witty chat assistant trained by [Your Company Name]. You maintain a friendly, engaging, and professional tone, focused on delivering high-quality, safe, and accurate conversational responses.",
        ),
        max_new_tokens: int = Input(description="Maximum tokens to generate.", default=1024),
        temperature: float = Input(description="Adjusts randomness of outputs. 0.0 is deterministic, 1.0 is highly random.", default=0.6, ge=0.0, le=5.0),
        top_p: float = Input(description="Nucleus sampling threshold.", default=0.9, ge=0.0, le=1.0),
    ) -> str:
        
        # 4. Apply Llama 3 Chat Template
        # This structure retains your IP via the system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        # Format the chat into the required Llama 3 token sequence: <|start_header_id|>...
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        # 5. Generate the response
        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            use_cache=True,
        )

        # 6. Decode and return the assistant's reply
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()