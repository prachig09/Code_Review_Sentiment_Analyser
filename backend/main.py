import gc
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Global model pointers
tokenizer = None
model = None

LABEL_MAPPING = {
    0: "Constructive / Positive",
    1: "Directly Harsh / Toxic",
    2: "Sarcastic / Passive-Aggressive"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    
    # Restrict PyTorch CPU thread count to save RAM overhead
    torch.set_num_threads(1)
    
    base_model_id = "distilbert-base-uncased"
    adapter_path = "./models/saved_sarcasm_lora_adapter"
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # Load base model using low CPU memory flag
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id, 
        num_labels=3,
        low_cpu_mem_usage=True
    )
    
    # Attach PEFT adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    # Force Garbage Collection after initialization
    gc.collect()
    
    yield
    
    # Cleanup on shutdown
    del model
    del tokenizer
    gc.collect()

app = FastAPI(
    title="Sarcasm & Developer Tone Analyzer API",
    lifespan=lifespan
)

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Sarcasm & Developer Tone API is running. Go to /docs to test endpoints."}

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