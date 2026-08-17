# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

app = FastAPI(title="Sarcasm & Developer Tone Analyzer API")

tokenizer = None
model = None

LABEL_MAPPING = {
    0: "Constructive / Positive",
    1: "Directly Harsh / Toxic",
    2: "Sarcastic / Passive-Aggressive"
}

@app.get("/")
def home():
    return {"message": "Sarcasm & Developer Tone API is running. Go to /docs to test endpoints."}

# In backend/main.py
@app.on_event("startup")
def load_model():
    global tokenizer, model
    base_model_id = "distilbert-base-uncased"
    
    # Updated relative path matching your repo structure
    adapter_path = "./models/saved_sarcasm_lora_adapter"
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_id, num_labels=3)
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_tone(payload: TextRequest):
    inputs = tokenizer(payload.text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).flatten().tolist()
        predicted_class = torch.argmax(logits, dim=1).item()
    
    return {
        "text": payload.text,
        "prediction": LABEL_MAPPING[predicted_class],
        "confidence": {
            "Positive": round(probs[0], 4),
            "Harsh": round(probs[1], 4),
            "Passive-Aggressive / Sarcastic": round(probs[2], 4)
        }
    }