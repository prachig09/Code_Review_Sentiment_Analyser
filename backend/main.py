import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime as ort

LABEL_MAPPING = {
    0: "Constructive / Positive",
    1: "Directly Harsh / Toxic",
    2: "Sarcastic / Passive-Aggressive"
}

app = FastAPI(title="Sarcasm & Developer Tone Analyzer API (ONNX)")

# Load tokenizer and ONNX inference session at boot
# This takes ~60MB RAM instead of 450MB+ with PyTorch
MODEL_DIR = "./models/onnx_light"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    session = ort.InferenceSession(f"{MODEL_DIR}/model.onnx")
except Exception as err:
    print(f"Failed to load ONNX model files: {err}")
    tokenizer = None
    session = None

class TextRequest(BaseModel):
    text: str

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

@app.get("/")
def home():
    return {"message": "Sarcasm & Developer Tone API (ONNX) is running. Send POST requests to /predict."}

@app.post("/predict")
def predict_tone(payload: TextRequest):
    if session is None or tokenizer is None:
        raise HTTPException(
            status_code=500, 
            detail="ONNX model or tokenizer is not initialized. Ensure model.onnx exists in ./models/onnx_light"
        )

    try:
        # Tokenize directly to NumPy arrays (no PyTorch tensors needed)
        inputs = tokenizer(
            payload.text, 
            return_tensors="np", 
            truncation=True, 
            max_length=128
        )
        
        # Format inputs for ONNX Runtime session
        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        
        # Run inference
        outputs = session.run(None, onnx_inputs)
        logits = outputs[0][0]
        
        # Calculate probabilities and prediction
        probs = softmax(logits).tolist()
        predicted_class = int(np.argmax(logits))
        
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