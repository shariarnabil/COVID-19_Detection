from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained model and label encoder
model = load_model('CNN_Covid19_Xray_Version.h5')  # Replace with your model path
le = pickle.load(open("models/Label_encoder.pkl", 'rb'))  # Load the label encoder

print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

# Path to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Convert and normalize
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 🔑 Resize to training input size (based on your model)
    image_resized = cv2.resize(image_rgb, (64, 64))  # or (128,128) if that was used in training

    image_normalized = image_resized / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)  # Shape: (1, 64, 64, 3)

    # Prediction
    predictions = model.predict(image_input)
    predicted_index = np.argmax(predictions)
    confidence_score = float(predictions[0][predicted_index])
    predicted_label = le.inverse_transform([predicted_index])[0]

    return predicted_label, confidence_score



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        try:
            predicted_label, confidence_score = process_image(file_path)
            return render_template(
                'result.html',
                image_path=file_path,
                filename=filename,
                predicted_label=predicted_label,
                confidence_score=round(confidence_score , 2)
            )
        except Exception as e:
            return f"Error processing image: {e}"


@app.route('/camera')
def camera():
    return render_template('camera.html')


if __name__ == '__main__':
    app.run(debug=True)
