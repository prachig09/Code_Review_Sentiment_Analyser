# convert_simple.py
import torch
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

base_model_id = "distilbert-base-uncased"
adapter_path = "./models/saved_sarcasm_lora_adapter"
output_dir = "./models/onnx_light"

os.makedirs(output_dir, exist_ok=True)

print("1. Loading and merging model weights...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
base_model = AutoModelForSequenceClassification.from_pretrained(base_model_id, num_labels=3)
model = PeftModel.from_pretrained(base_model, adapter_path)

# Merge LoRA weights into base model
merged_model = model.merge_and_unload()
merged_model.eval()

# Save tokenizer to the new directory
tokenizer.save_pretrained(output_dir)

print("2. Exporting to ONNX...")
dummy_input = tokenizer("Test input text for ONNX export", return_tensors="pt")

torch.onnx.export(
    merged_model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    f"{output_dir}/model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    },
    opset_version=14
)

print(f"Done! Model saved to {output_dir}/model.onnx")