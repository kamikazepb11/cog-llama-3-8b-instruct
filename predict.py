import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from cog import BasePredictor, Input
from typing import Iterator # CORRECTED IMPORT: Iterator is now imported from typing, not cog

# The Llama 3 prompt template is required for correct dialogue formatting
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# The official Hugging Face ID for the Llama 3 8B Instruct model
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

class Predictor(BasePredictor):
    def setup(self):
        """
        Load the model and tokenizer into memory. This runs once on startup.
        We use 4-bit quantization (QLoRA) to reduce VRAM usage.
        """
        print("Loading model...")
        
        # 1. Define 4-bit quantization configuration
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # 2. Load the model and tokenizer using the configuration
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=self.bnb_config,
            device_map="auto", # Automatically maps model layers to available hardware
            trust_remote_code=True
        )
        print("Model loaded successfully.")

    def predict(
        self,
        prompt: str = Input(
            description="The prompt to send to the model for generation.",
            default="Explain why large language models are so resource intensive in one paragraph."
        ),
        system_prompt: str = Input(
            description="System instructions to set the model's behavior/persona.",
            default="You are a helpful and detailed assistant."
        ),
        max_new_tokens: int = Input(
            description="The maximum number of tokens to generate in the response.",
            default=512,
            ge=1,
            le=4096
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs. Higher values are more random.",
            default=0.6,
            ge=0.1,
            le=5.0
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percent of most likely tokens. Lower values are more deterministic.",
            default=0.9,
            ge=0.01,
            le=1.0
        ),
    ) -> Iterator[str]:
        """
        Run a single prediction and stream the output.
        """
        # 1. Format the conversation using the Llama 3 chat template
        if system_prompt:
            # Use the official Llama 3 template format
            formatted_prompt = f"{B_INST}{B_SYS}{system_prompt}{E_SYS}{prompt}{E_INST}"
        else:
            formatted_prompt = f"{B_INST} {prompt} {E_INST}"

        # 2. Tokenize the input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        # 3. Define termination tokens for Llama 3
        # Llama 3 uses <|eot_id|> (128009) and the standard eos_token_id
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # 4. Start the streaming generation
        output_stream = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=terminators,
            repetition_penalty=1.1,
            streamer=True, # Enable streaming output
        )
        
        # 5. Yield tokens from the stream
        for token in output_stream:
            # The streamer will output token objects, yield the decoded string
            yield self.tokenizer.decode([token], skip_special_tokens=True)
