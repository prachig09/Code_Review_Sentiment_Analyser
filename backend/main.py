import gc
import os
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

LABEL_MAPPING = {
    0: "Constructive / Positive",
    1: "Directly Harsh / Toxic",
    2: "Sarcastic / Passive-Aggressive"
}

# Global pointers initialized as None
tokenizer = None
model = None

def get_model_and_tokenizer():
    """Dynamically loads model into memory with half-precision (FP16) on demand."""
    global tokenizer, model
    
    if model is None or tokenizer is None:
        # 1. Restrict PyTorch thread count to lower system memory overhead
        torch.set_num_threads(1)
        
        base_model_id = "distilbert-base-uncased"
        adapter_path = "./models/saved_sarcasm_lora_adapter"
        
        # 2. Fetch optional HF_TOKEN if set in environment
        hf_token = os.getenv("HF_TOKEN", None)
        
        # 3. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_path, 
            token=hf_token
        )
        
        # 4. Load Base Model in float16 (half-precision) to halve memory footprint
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_id, 
            num_labels=3,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            token=hf_token
        )
        
        # 5. Attach PEFT LoRA adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        
        # 6. Force garbage collection to sweep temporary allocation artifacts
        gc.collect()
        
    return model, tokenizer

@asynccontextmanager
async def lifespan(app: FastAPI):
    # App startup logic (kept lightweight to prevent OOM on server boot)
    yield
    # Cleanup on server shutdown
    global model, tokenizer
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
    return {"message": "Sarcasm & Developer Tone API is running. Send POST requests to /predict."}

@app.post("/predict")
def predict_tone(payload: TextRequest):
    try:
        # Load model lazily on first incoming request
        active_model, active_tokenizer = get_model_and_tokenizer()
        
        inputs = active_tokenizer(
            payload.text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128
        )
        
        with torch.no_grad():
            outputs = active_model(**inputs)
            logits = outputs.logits
            # Convert FP16 logits back to FP32 for numeric precision during softmax
            probs = torch.softmax(logits.to(torch.float32), dim=1).flatten().tolist()
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))