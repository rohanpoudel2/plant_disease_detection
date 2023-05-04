import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os

# Define the model path and image size
MODEL_PATH = "model.tflite"
IMAGE_SIZE = (299, 299)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Create a list of the class names in the correct order
classes = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy", "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
           "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"]

# Define the function to preprocess the image


def preprocess_image(image):
    img = cv2.resize(image, IMAGE_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# Define the function to make predictions on an image

def predict(image_path_or_pil_image):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess image
    if isinstance(image_path_or_pil_image, str):
        img = cv2.imread(image_path_or_pil_image)
        img = preprocess_image(img, input_details[0]['shape'][1:3])
    else:
        img = np.array(image_path_or_pil_image)
        img = preprocess_image(img)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Convert the output tensor to predictions
    if isinstance(output_data, list):
        results = {}
        for label, confidence in zip(classes, output_data):
            results[label] = confidence
    else:
        predictions = output_data.squeeze()
        top_k = np.argsort(predictions)[::-1][:5]
        results = {}
        for idx in top_k:
            results[classes[idx]] = float(predictions[idx])

    return results


# Define the Gradio interface
inputs = gr.inputs.Image()
outputs = gr.outputs.Label(num_top_classes=5)
interface = gr.Interface(fn=predict, inputs=inputs,
                         outputs=outputs, capture_session=True)

# Run the interface
interface.launch()
