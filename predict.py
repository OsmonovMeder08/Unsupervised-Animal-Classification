import joblib
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_SIZE = (128, 128)

def load_model():
    try:
        base_model = joblib.load('resnet_model.joblib')
        umap_model = joblib.load('umap_model.joblib')
        kmeans = joblib.load('kmeans_model.joblib')
        return base_model, umap_model, kmeans
    except FileNotFoundError as e:
        print(f"ERROR: Model files not found. Please run train.py first.")
        print(f"Missing: {e}")
        exit(1)

def predict_cluster(img_path, base_model, umap_model, kmeans):
    """Predict cluster for a single image"""
    try:
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = preprocess_input(img_to_array(img)[np.newaxis, ...])

        features = base_model.predict(img_array, verbose=0)
        features_umap = umap_model.transform(features)
        cluster = kmeans.predict(features_umap)[0]

        return cluster
    except Exception as e:
        raise Exception(f"Failed to process image: {str(e)}")

def validate_input(user_input):
    """Validate user input"""
    if not user_input.strip():
        raise ValueError("Input cannot be empty")

    if not os.path.isfile(user_input):
        raise FileNotFoundError(f"File does not exist: {user_input}")

    if not user_input.lower().endswith(('.jpg', '.png', '.jpeg')):
        raise ValueError("File must be an image (.jpg, .png, .jpeg)")

    return user_input

print("=" * 60)
print("STAGE 2: ANIMAL CLUSTER PREDICTION")
print("=" * 60)

print("\nLoading models...")
base_model, umap_model, kmeans = load_model()
print("✓ Models loaded successfully\n")

while True:
    try:
        user_input = input("Enter image path (or 'quit' to exit): ").strip()

        if user_input.lower() == 'quit':
            print("\nGoodbye! 👋")
            break

        img_path = validate_input(user_input)

        print(f"Processing {img_path}...")
        cluster = predict_cluster(img_path, base_model, umap_model, kmeans)

        print(f"✓ Prediction: Cluster {cluster}")
        print()

    except FileNotFoundError as e:
        print(f"❌ File Error: {e}\n")
    except ValueError as e:
        print(f"❌ Input Error: {e}\n")
    except Exception as e:
        print(f"❌ Prediction Error: {e}\n")
