# app_ui.py
import gradio as gr
import requests

FASTAPI_URL = "http://127.0.0.1:8000/predict"



def analyze_tone(user_text):
    if not user_text.strip():
        return "Please enter text.", {}
    
    try:
        response = requests.post(FASTAPI_URL, json={"text": user_text}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data["prediction"], data["confidence"]
        return f"Error: Status {response.status_code}", {}
    except Exception as e:
        return f"Could not connect to FastAPI server: {str(e)}", {}

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Code Review Sarcasm & Passive-Aggression Detector")
    gr.Markdown("Detect hidden passive-aggression or sarcasm in developer feedback using a LoRA fine-tuned DistilBERT model.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                lines=4, 
                placeholder="e.g., Wow, thanks for breaking the build right before Friday release!", 
                label="PR Comment or Code Feedback"
            )
            submit_btn = gr.Button("Analyze Tone", variant="primary")
            
            # Preset examples for quick testing
            gr.Examples(
                examples=[
                    ["Per my previous comment, this was already documented."],
                    ["Wow, incredible work testing this in production."],
                    ["Great PR! Very clean implementation and easy to follow."],
                    ["This is atrocious. Delete this and start over."]
                ],
                inputs=text_input
            )
        
        with gr.Column():
            output_label = gr.Textbox(label="Detected Tone")
            confidence_output = gr.Label(label="Probability Distribution")

    submit_btn.click(
        fn=analyze_tone, 
        inputs=[text_input], 
        outputs=[output_label, confidence_output]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)