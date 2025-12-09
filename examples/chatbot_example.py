#!/usr/bin/env python3

"""
Minimal inference example for SmolLM2-135M-Instruct.

Filepath: ./examples/chatbot_example.py
Project: CPEN455-Project-2025W1
Description: Loads the model and runs a single inference.

Usage:
    uv run -m examples.chatbot_example
"""

import os
import pdb
from dotenv import load_dotenv
import pandas as pd

from model import LlamaModel
from utils.sample import sample
from utils.weight_utils import load_model_weights
from model.config import Config
from model.tokenizer import Tokenizer
from utils.download import _resolve_snapshot_path
from utils.device import set_device

if __name__ == "__main__":
    load_dotenv()
    
    checkpoint = os.getenv("MODEL_CHECKPOINT")
    model_cache_dir = os.getenv("MODEL_CACHE_DIR")

    # Set device to GPU if available, to MPS if on Mac with M-series chip, else CPU
    device = set_device()

    # Tokenizer and config loading now automatically download if not cached
    tokenizer = Tokenizer.from_pretrained(checkpoint, cache_dir=model_cache_dir)
    
    base_path = _resolve_snapshot_path(checkpoint, cache_dir=model_cache_dir)
    config = Config._find_config_files(base_path)

    model = LlamaModel(config)

    load_model_weights(model, checkpoint, cache_dir=model_cache_dir, device=device)

    model = model.to(device)
    model.eval()


    emails = pd.read_csv("autograder/cpen455_released_datasets/train_val_subset.csv")
    

    for i in range(10):
        email = emails["Message"][i]
        truth = str(emails["Spam/Ham"][i])

        messages = [ 
                    {"role": "user", "content": f"Classify this email as 'SPAM' or 'HAM'. Return only the classifier 'SPAM' or 'HAM'. Output only ONE word. The email is:\n\n---{email}---"}
        ]

        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        #print(input_text)
        
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
        input_len = inputs.shape[1]
        outputs = sample(
            model,
            inputs,
            max_new_tokens=10,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
        )
        predicted_text = tokenizer.decode(outputs[0][input_len:],skip_special_tokens=True).strip()
        truth_label = 'SPAM' if truth == '1' else 'HAM' 
        print(
            f"PREDICTION: {predicted_text} "
            f"TRUTH: {truth_label}\n"
        )