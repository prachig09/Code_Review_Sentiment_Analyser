import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType

# 1.  Dataset
dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

# 2. Mapping multi-label list to a single target label
def map_emotions(example):
    labels_list = example["labels"]
    # 27 is 'sarcasm' in go_emotions
    if 27 in labels_list:
        target_label = 2  # Sarcastic / Passive-Aggressive
    elif any(l in labels_list for l in [10, 11, 25]):  # disapproval, disgust, sadness
        target_label = 1  # Harsh / Negative
    else:
        target_label = 0  # Constructive / Positive
    
    return {"label": target_label}

# Apply remapping
dataset = dataset.map(map_emotions)

# 3. Tokenizer & Preprocessing
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_func(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

# Tokenize and remove original columns that cause collator conflicts
tokenized_dataset = dataset.map(
    tokenize_func, 
    batched=True, 
    remove_columns=["text", "labels", "id"]
)

# 4. Loading Base Model for 3 Classes
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 5. Configure PEFT / LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"]
)

model = get_peft_model(model, peft_config)

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir="./sarcasm_lora",
    learning_rate=3e-4,
    per_device_train_batch_size=32,
    num_train_epochs=3,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_steps=50
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 7. Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].select(range(8000)),
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
)

# 8. Start Training
trainer.train()

# 9. Save Adapter Checkpoint
model.save_pretrained("./saved_sarcasm_lora_adapter")
tokenizer.save_pretrained("./saved_sarcasm_lora_adapter")
print("Sarcasm & Passive-Aggressive PEFT model saved successfully!")
