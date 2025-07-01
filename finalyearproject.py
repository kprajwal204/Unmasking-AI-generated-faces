import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import gradio as gr
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from io import BytesIO
import plotly.graph_objects as go
from PIL import Image as PILImage

# Load pre-trained models
cnn_model = tf.keras.models.load_model("cnn_model.h5")
inception_model = tf.keras.models.load_model("inception_model_trained.h5")
vgg16_model = tf.keras.models.load_model("vgg16_model_trained.h5")
xception_model = tf.keras.models.load_model("xception_model_trained.h5")

models = {
    "CNN": cnn_model,
    "InceptionV3": inception_model,
    "VGG16": vgg16_model,
    "Xception": xception_model
}

MODEL_INFO = {
    "CNN": "Custom CNN: Lightweight and fast.",
    "InceptionV3": "InceptionV3: Excels at complex patterns.",
    "VGG16": "VGG16: Deep and reliable.",
    "Xception": "Xception: Detailed and accurate."
}

# Preprocess image based on each model's expected input size
def preprocess_image(model, img):
    target_size = model.input_shape[1:3]
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Generate combined bar chart for all models
def generate_combined_chart(results):
    labels = list(results.keys())
    values = [confidence * 100 for confidence in results.values()]
    colors = ['green' if v < 50 else 'red' for v in values]

    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))
    fig.update_layout(title="Confidence by Model", yaxis=dict(range=[0, 100]), height=300)
    img_bytes = fig.to_image(format="png")
    return PILImage.open(BytesIO(img_bytes))

# Prediction logic with ensemble results
def predict_all(image_file):
    yield "Processing...", None, None

    if not image_file:
        yield "Upload an image to proceed.", None, None
        return

    img = image.load_img(image_file)
    final_results = {}
    votes = {"Real": 0, "AI-generated": 0}

    for name, model in models.items():
        try:
            img_array = preprocess_image(model, img)
            prediction = model.predict(img_array).item()
            label = "AI-generated" if prediction > 0.5 else "Real"
            confidence = prediction if label == "AI-generated" else 1 - prediction
            final_results[name] = confidence
            votes[label] += 1
        except Exception:
            final_results[name] = 0.0

    majority_vote = "AI-generated" if votes["AI-generated"] > votes["Real"] else "Real"
    average_conf = np.mean(list(final_results.values()))
    result = f"Final verdict: {majority_vote} with avg confidence {average_conf*100:.2f}%"
    chart_img = generate_combined_chart(final_results)

    yield result, chart_img, img

# Reset UI elements
def reset():
    return None, None, None, None

# Gradio Interface Setup
def gradio_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="Real vs AI Image Detector") as interface:
        gr.Markdown("# 🧐 Real vs AI-Generated Image Detection")
        gr.Markdown("Upload an image to get predictions from four powerful models.")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Upload Image")
                with gr.Row():
                    submit_btn = gr.Button("Analyze", variant="primary")
                    reset_btn = gr.Button("Reset", variant="secondary")
            with gr.Column(scale=2):
                preview = gr.Image(label="Image Preview")
                result = gr.Textbox(label="Prediction", interactive=False)
                chart = gr.Image(label="Model Confidence")

        submit_btn.click(
            fn=predict_all,
            inputs=[image_input],
            outputs=[result, chart, preview]
        )

        reset_btn.click(
            fn=reset,
            inputs=None,
            outputs=[image_input, preview, result, chart]
        )

    # Launch directly in browser without showing any links
    interface.launch(inbrowser=True, show_error=True)

# Run the app
gradio_interface()
