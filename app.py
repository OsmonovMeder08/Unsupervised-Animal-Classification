from flask import Flask, render_template, request, jsonify, send_file
import joblib
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

IMG_SIZE = (128, 128)
SAMPLES_TO_SHOW = 10

models_cache = {}

def load_models():
    global models_cache
    if not models_cache:
        try:
            models_cache['base_model'] = joblib.load('resnet_model.joblib')
            models_cache['umap_model'] = joblib.load('umap_model.joblib')
            models_cache['kmeans'] = joblib.load('kmeans_model.joblib')
            with open('metadata.json', 'r') as f:
                models_cache['metadata'] = json.load(f)
        except Exception as e:
            print(f"Error loading models: {type(e).__name__}: {e}")
            return False
    return True

def get_model_metrics():
    """Get model performance metrics"""
    if not load_models():
        return None
    return models_cache['metadata']

def predict_image(img_path):
    """Predict cluster for an image"""
    if not load_models():
        return None

    try:
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = preprocess_input(img_to_array(img)[np.newaxis, ...])
        features = models_cache['base_model'].predict(img_array, verbose=0)
        features_umap = models_cache['umap_model'].transform(features)
        cluster = models_cache['kmeans'].predict(features_umap)[0]
        return int(cluster)
    except Exception as e:
        return None

def get_dataset_samples():
    """Get sample images from dataset"""
    data_dir = 'data/dataset/'
    if not os.path.exists(data_dir):
        return []

    files = sorted([f for f in os.listdir(data_dir)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))])[:SAMPLES_TO_SHOW]

    samples = []
    for file in files:
        file_path = os.path.join(data_dir, file)
        try:
            # Read image and convert to base64
            with open(file_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            samples.append({
                'filename': file,
                'path': file_path,
                'data': img_data
            })
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return samples

@app.route('/')
def index():
    """Main page"""
    if not load_models():
        return '''
        <h1>Error</h1>
        <p>Models not found. Please run train.py first.</p>
        ''', 500

    metrics = get_model_metrics()
    samples = get_dataset_samples()

    return render_template('index.html',
                         metrics=metrics,
                         samples=samples,
                         num_samples_shown=len(samples))

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        if not os.path.isdir('uploads'):
            os.mkdir('uploads')

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        cluster = predict_image(filepath)

        if cluster is None:
            return jsonify({'error': 'Failed to process image'}), 400

        return jsonify({
            'success': True,
            'cluster': cluster,
            'filename': file.filename
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualization')
def visualization():
    """Get cluster visualization"""
    if not load_models():
        return jsonify({'error': 'Models not loaded'}), 500

    try:
        data_dir = 'data/dataset/'
        images = []
        filenames = []
        max_images = 500

        for file in sorted(os.listdir(data_dir)):
            if len(images) >= max_images:
                break
            path = os.path.join(data_dir, file)
            if os.path.isfile(path) and file.lower().endswith(('.jpg', '.png', '.jpeg')):
                try:
                    img = load_img(path, target_size=IMG_SIZE)
                    images.append(preprocess_input(img_to_array(img)))
                    filenames.append(file)
                except Exception as img_err:
                    print(f"Error loading {file}: {img_err}")

        if len(images) == 0:
            return jsonify({'error': 'No images found'}), 400

        print(f"Loaded {len(images)} images")
        images = np.array(images)
        print(f"Images shape: {images.shape}")

        features = models_cache['base_model'].predict(images, verbose=0)
        print(f"Features shape: {features.shape}")

        features_umap = models_cache['umap_model'].transform(features)
        print(f"UMAP features shape: {features_umap.shape}")

        labels = models_cache['kmeans'].predict(features_umap)
        pca_2d = PCA(n_components=2, random_state=42)
        embedding_2d = pca_2d.fit_transform(features_umap)
        print(f"2D embedding shape: {embedding_2d.shape}")
        data = {
            'x': embedding_2d[:, 0].tolist(),
            'y': embedding_2d[:, 1].tolist(),
            'clusters': labels.tolist(),
            'num_clusters': int(models_cache['metadata']['num_clusters'])
        }

        return jsonify(data)

    except Exception as e:
        print(f"Visualization error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("STAGE 3: WEB INTERFACE FOR ANIMAL CLASSIFICATION")
    print("=" * 60)
    print("\nServer starting at http://localhost:5500")
    print("Access the web interface in your browser\n")
    app.run(debug=True, port=5500)
