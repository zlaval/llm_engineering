import gradio as gr
from transformers import ViTImageProcessor, ViTForImageClassification

MODEL_NAME = 'google/vit-base-patch16-224'

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
model = ViTForImageClassification.from_pretrained(MODEL_NAME)

def classify_image(img):
     inputs = processor(images=img, return_tensors="pt")
     outputs = model(**inputs)
     logits = outputs.logits
     predicted_class_idx = logits.argmax(-1).item()
     return model.config.id2label[predicted_class_idx]

with gr.Blocks() as ui:
    with gr.Row(equal_height=True):
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
        with gr.Column(elem_classes="center"):
            with gr.Row():
                classify_btn = gr.Button("Classify")
        with gr.Column():
            output_text = gr.Textbox(label="Prediction")
    classify_btn.click(fn=classify_image, inputs=image_input, outputs=output_text)

ui.launch()
