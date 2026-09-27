# backend/convert-to-onnx.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

ADAPTER_PATH = "./models/saved_sarcasm_lora_adapter"
BASE_MODEL_ID = "distilbert-base-uncased"
OUTPUT_DIR = "./models/onnx_light"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("1. Loading base model and attaching LoRA weights...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_ID, num_labels=3)
lora_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("2. Merging LoRA layers into base model...")
merged_model = lora_model.merge_and_unload()
merged_model.eval()

# Save tokenizer files
tokenizer.save_pretrained(OUTPUT_DIR)

print("3. Exporting to single ONNX file using legacy tracer...")
dummy_input = tokenizer("Sample text for ONNX export tracing", return_tensors="pt")

onnx_path = os.path.join(OUTPUT_DIR, "model.onnx")

# Remove split data file if it exists
data_file = os.path.join(OUTPUT_DIR, "model.onnx.data")
if os.path.exists(data_file):
    os.remove(data_file)

# Export using legacy TorchScript backend (bypasses PyTorch Dynamo)
torch.onnx.export(
    merged_model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    onnx_path,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    },
    dynamo=False  # Crucial for PyTorch 2.x on Python 3.13!
)

print(f"Success! Model successfully exported to: {onnx_path}")