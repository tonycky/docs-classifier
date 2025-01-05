from flask import Flask, request, render_template, jsonify
from flask_cors import CORS  # Import the CORS module
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import io
import fitz  # PyMuPDF
from collections import Counter
import threading

app = Flask(__name__)

# Disable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for all routes

# Force the script to use CPU
device = "cpu"
print(f"Using device: {device}")

# Load the CLIP model and processor
model_name = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# Global variable to track if a request is being processed
request_lock = threading.Lock()
is_processing = False

@app.route('/', methods=['GET', 'POST'])
def index():
    global is_processing

    if request.method == 'POST':
        # Check if another request is being processed
        if is_processing:
            return jsonify(error="Server is busy processing another request. Please try again later."), 429  # 429: Too Many Requests

        # Acquire the lock to indicate that a request is being processed
        with request_lock:
            is_processing = True

            try:
                # Get the uploaded file
                file = request.files['file']
                
                # Get the labels from the form
                labels = request.form.get('labels').split('"')  # Assuming labels are comma-separated
                
                # Remove empty strings from the labels list
                labels = [label.strip() for label in labels if label.strip()]

                # Remove "," from the label list
                labels = [label for label in labels if label != ","]
                
                # Read the file
                file_content = file.read()
                
                # Check if the file is a PDF
                if file.filename.endswith('.pdf'):
                    # Convert PDF to images
                    images = pdf_to_images(file_content)
                    
                    # Classify each image and aggregate results
                    all_scores = []
                    for image in images:
                        scores = classify_image(image, labels)
                        all_scores.append(scores)
                    
                    # Average scores across all pages
                    avg_scores = {label: sum(page_scores[label] for page_scores in all_scores) / len(all_scores) 
                                  for label in labels}
                    
                    # Determine the most common label for the entire PDF
                    most_common_label = max(avg_scores, key=avg_scores.get)
                    
                    # Return the result as a JSON response
                    return jsonify(result=most_common_label, scores=avg_scores, model_name=model_name)
                
                else:
                    # Assume the file is an image
                    image = Image.open(io.BytesIO(file_content))
                    
                    # Classify the image
                    scores = classify_image(image, labels)
                    most_common_label = max(scores, key=scores.get)
                    
                    # Return the result as a JSON response
                    return jsonify(result=most_common_label, scores=scores, model_name=model_name)
            
            finally:
                # Release the lock and reset the processing flag
                is_processing = False

    return render_template('index.html')

def pdf_to_images(pdf_content):
    """
    Convert a PDF file to a list of images.
    """
    images = []
    pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        images.append(img)
    
    return images

def classify_image(image, labels):
    """
    Classify an image using CLIP and a set of labels.
    Returns a dictionary of labels and their corresponding probabilities.
    """
    # Prepare the inputs
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)

    # Get model outputs
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # Image-text similarity scores
    probs = logits_per_image.softmax(dim=1).detach().cpu().numpy()  # Convert to probabilities

    # Create a dictionary of labels and their probabilities
    scores = {label: float(probs[0][i]) for i, label in enumerate(labels)}
    
    return scores

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)