# run_quantize.py
import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

input_model = "models/onnx_light/model.onnx"
output_model = "models/onnx_light/model_quantized.onnx"

print("1. Quantizing 256MB model to INT8...")
quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    weight_type=QuantType.QUInt8
)

print(f"2. Quantization complete! Saved to {output_model}")

# Clean up original large model.onnx so we don't keep redundant 256MB on disk
if os.path.exists(output_model):
    os.remove(input_model)
    os.rename(output_model, input_model)
    print("3. Replaced original model.onnx with the quantized model!")