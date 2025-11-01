import os
import replicate
import json
from datetime import datetime

# --- CRITICAL CONFIGURATION ---
# We use the manually created model name for both source and destination.
# This should resolve the 404 error now that the model registry exists.
TRAINER_VERSION_REF = "resonance/svs" 

# The desired final model name (The registry that will hold the fine-tune).
DESTINATION_MODEL = "resonance/svs" 

# The direct, publicly accessible URL for your training data from Google Drive.
TRAINING_DATA_URL = "https://drive.google.com/uc?export=download&id=1s26AGX9C1VEdfkzbuaJs1hXGNO3rPnVA" 

# --- Hyperparameters ---
HYPERPARAMETERS = {
    # NOTE: Replicate will automatically select the Llama 3 8B model when using this trainer.
    "train_data": TRAINING_DATA_URL,
    "lora_rank": 4,           
    "lora_alpha": 8,          
    "num_train_epochs": 3,    
    "max_steps": -1,          
    "learning_rate": 2e-05,   
    "max_length": 2048        
}

def launch_training():
    """Launches the fine-tuning job using the official Llama 3 trainer template."""
    
    # Check for API Token
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        print("FATAL: REPLICATE_API_TOKEN environment variable not set. Please set the secret in GitHub.")
        return
        
    # 1. Initialize Replicate Client
    try:
        client = replicate.Client(api_token=api_token)
    except Exception:
        print("FATAL: Could not initialize Replicate client.")
        return

    # 2. Launch the training job
    print("\n--- Attempting to Initiate Fine-Tuning Job ---")
    print(f"Base Trainer Version (Source): {TRAINER_VERSION_REF}")
    print(f"Destination Model: {DESTINATION_MODEL}")
    print(f"Training Data URL: {TRAINING_DATA_URL}")
    print(f"Parameters: {HYPERPARAMETERS}")

    try:
        # We use your model name for both version and destination.
        # This is the last and most direct API call possible.
        training = client.trainings.create(
            version=TRAINER_VERSION_REF,  
            destination=DESTINATION_MODEL, 
            input=HYPERPARAMETERS,
        )
        
        # 3. Print success details
        print("\n--- SUCCESSFULLY INITIATED TRAINING JOB ---")
        print(f"Training ID: {training.id}")
        print(f"Status: {training.status}")
        print(f"Monitor URL: https://replicate.com/p/{training.id}")
        print("------------------------------------------\n")

    except replicate.exceptions.ReplicateError as e:
        print("\n--- ERROR INITIATING TRAINING JOB ---")
        print(f"An error occurred: ReplicateError Details:")
        print(e)
        print("------------------------------------------\n")
        

if __name__ == "__main__":
    launch_training()