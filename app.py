import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

app = Flask(__name__)

# 1. Rebuild the exact empty architecture from Colab
print("Building model architecture...")
base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(25, activation='softmax')
])

# 2. Load JUST the learned knowledge (weights) into the empty model
try:
    # IMPORTANT: change '.h5' to '.keras' if your friend sent the keras version
    model.load_weights('malvision_model.h5') 
    print("✅ Model weights loaded successfully!")
except Exception as e:
    print(f"❌ Warning: Model not loaded. {e}")


# 2. Define your 25 Malimg classes (You will get this exact list from Colab)
CLASSES = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 
           'Autorun.K', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A', 
           'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 
           'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 
           'Swizzor.gen!E', 'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A']

def convert_and_preprocess(file_bytes):
    """Converts raw bytes to a 224x224 RGB image array for VGG16"""
    # Convert bytes to 1D numpy array
    d = np.frombuffer(file_bytes, dtype=np.uint8)
    
    # Determine width based on file size
    file_size_kb = len(d) / 1024
    if file_size_kb < 10: width = 32
    elif file_size_kb < 30: width = 64
    elif file_size_kb < 60: width = 128
    elif file_size_kb < 100: width = 256
    elif file_size_kb < 200: width = 384
    elif file_size_kb < 500: width = 512
    elif file_size_kb < 1000: width = 768
    else: width = 1024
    
    height = int(len(d) / width)
    if height == 0: height = 1 # Fallback for tiny files
    
    d = d[:height * width]
    img_matrix = np.reshape(d, (height, width))
    
    # Create image, resize to 224x224, and convert to RGB (3 channels)
    img = Image.fromarray(img_matrix).resize((224, 224))
    img = img.convert("RGB")
    
    # Normalize pixel values to [0, 1] and expand dimensions for the model
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Read the file and process it
        file_bytes = file.read()
        processed_img = convert_and_preprocess(file_bytes)
        
        # Run Inference
        predictions = model.predict(processed_img)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) * 100
        
        result = {
            'family': CLASSES[predicted_class_idx],
            'confidence': round(confidence, 2)
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)