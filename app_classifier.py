from flask import Flask, request, render_template, jsonify
from flask_cors import CORS  # Import the CORS module

import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import io

app = Flask(__name__)

# Disable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for all routes


# Force the script to use CPU
device = "cpu"
print(f"Using device: {device}")

# Load the CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        
        # Get the labels from the form
        labels = request.form.get('labels').split('"')

        labels = request.form.get('labels').split('"')
        
        # Read the image file
        image = Image.open(io.BytesIO(file.read()))
        
        # Classify the image
        result = classify_image(image, labels)
        
        # Return the result as a JSON response
        return jsonify(result=result)
    
    return render_template('index.html')

def classify_image(image, labels):
    """
    Classify an image using CLIP and a set of labels.
    """
    # Prepare the inputs
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)

    # Get model outputs
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # Image-text similarity scores
    probs = logits_per_image.softmax(dim=1).detach().cpu().numpy()  # Convert to probabilities

    # Get the label with the highest probability
    max_prob_index = probs.argmax()
    result_label = labels[max_prob_index]
    
    return result_label

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0',port=5001)