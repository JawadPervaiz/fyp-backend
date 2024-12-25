from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = 'best_model.keras'  # Adjust path as needed
model = load_model(MODEL_PATH)

# Function to preprocess image for prediction
def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess the input image for the model.
    Resize the image and normalize pixel values.
    """
    img = load_img(image_path, target_size=target_size)  # Resize image
    img = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Function to count buildings from the model's output
def count_buildings(prediction, threshold=0.5):
    """
    Count the number of buildings from the prediction mask.
    Uses thresholding and connected components analysis.
    """
    # Convert prediction to binary mask
    binary_mask = (prediction > threshold).astype(np.uint8)

    # Morphological operations to clean noise
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Count connected components (regions)
    num_labels, _ = cv2.connectedComponents(binary_mask)
    return num_labels - 1  # Subtract 1 for the background label

# Root endpoint to verify API status
@app.route('/')
def index():
    """
    Health check endpoint to verify the API is running.
    """
    return "API is working"

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts an image file, processes it, and returns the building count.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file temporarily
    temp_path = 'temp_uploaded_image.jpg'
    file.save(temp_path)

    try:
        # Preprocess the uploaded image
        preprocessed_image = preprocess_image(temp_path)

        # Predict using the model
        prediction = model.predict(preprocessed_image)[0]  # Remove batch dimension

        # Count buildings in the predicted mask
        building_count = count_buildings(prediction)

        # Cleanup temporary file
        os.remove(temp_path)

        # Return the response
        return jsonify({
            'building_count': building_count,
            'message': 'Prediction successful'
        })
    except Exception as e:
        # Cleanup temporary file in case of an error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)
