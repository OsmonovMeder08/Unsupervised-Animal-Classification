# 🦁 Unsupervised Animal Classification Project

A complete 3-stage machine learning project for clustering animal images using unsupervised learning.



├── train.py              # Stage 1: Train the model
├── predict.py            # Stage 2: Make predictions (CLI)
├── app.py                # Stage 3: Web interface
├── templates/
│   └── index.html        # Web interface UI
├── data/
│   └── dataset/          # Animal images (JPG, PNG, JPEG)
├── outputs/              # Generated visualizations
├── uploads/              # Uploaded images for prediction
├── kmeans_model.joblib   # KMeans clustering model
├── umap_model.joblib     # UMAP dimensionality reduction model
├── resnet_model.joblib   # ResNet50 feature extraction model
├── metadata.json         # Model metadata and metrics
└── requirements.txt      # Python dependencies
```

## Installation

1. **Create a virtual environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Stage 1: Train the Model 🏋️

Train the clustering model on your animal image dataset:

```bash
python train.py
```

**What it does:**
- Loads all images from `data/dataset/`
- Extracts features using ResNet50 (pre-trained on ImageNet)
- Reduces dimensionality using UMAP
- Clusters images using KMeans (5 clusters by default)
- Displays clustering metrics (Silhouette Score)
- Saves models and visualization

**Output:**
- `kmeans_model.joblib` - KMeans clustering model
- `umap_model.joblib` - UMAP dimensionality reduction
- `resnet_model.joblib` - ResNet50 feature extractor
- `metadata.json` - Model metrics
- `outputs/animal_clusters.png` - Cluster visualization

**Model Accuracy:**
- **Silhouette Score**: -1 to 1 (higher is better, >0.5 is good)
- This metric measures how well-separated the clusters are

**Tips for Improvement:**
- Use more images (at least 100+)
- Adjust `NUM_CLUSTERS` in train.py (typically 3-10)
- Check if all images are animal images and similar quality


### Stage 2: Make Predictions 🎯

Use the trained model to classify new animal images:

```bash
python predict.py
```

**Features:**
- ✓ Interactive terminal interface
- ✓ Input validation (checks file exists, is an image)
- ✓ Error handling (clear error messages)
- ✓ Loop until user exits (type 'quit' to exit)

**Example:**
```
Enter image path (or 'quit' to exit): data/dataset/3201.jpg
Processing data/dataset/3201.jpg...
✓ Prediction: Cluster 2

Enter image path (or 'quit' to exit): invalid.jpg
❌ File Error: File does not exist: invalid.jpg

Enter image path (or 'quit' to exit): quit
Goodbye! 👋
```

**Input Validation:**
- Checks if file exists
- Verifies it's an image (.jpg, .png, .jpeg)
- Handles missing files gracefully
- Catches any processing errors


### Stage 3: Web Interface 🌐

Launch the interactive web dashboard:

```bash
python app.py
```

Then open http://localhost:5000 in your browser

**Features:**
- 📊 **Model Metrics**: Silhouette score, cluster count, sample count
- 🔍 **Architecture Info**: Shows the ML pipeline used
- 🎯 **Prediction Form**: Upload images → Get cluster predictions
- 📈 **Interactive Visualization**: 2D cluster plot with Plotly
- 🖼️ **Dataset Samples**: View 10 sample images from dataset
- 📱 **Responsive Design**: Works on desktop and mobile
- ⚡ **Real-time Predictions**: Instant results after upload

**How to Use Web Interface:**
1. Open http://localhost:5000
2. View model metrics and architecture
3. Scroll to "Make a Prediction" section
4. Click upload area or drag & drop an image
5. See the cluster prediction result
6. View interactive cluster visualization
7. See sample images from the dataset


## Model Architecture 🏗️

```
Image Input (224×224)
    ↓
ResNet50 (Feature Extraction)
- Pre-trained on ImageNet
- Extracts 2048-dimensional features
    ↓
UMAP (Dimensionality Reduction)
- Reduces to 50 dimensions
- Preserves local structure
    ↓
KMeans (Clustering)
- Groups similar images
- 5 clusters (configurable)
    ↓
Cluster Label (0-4)
```

### Why This Architecture?

1. **ResNet50**: Powerful pre-trained model for image understanding
2. **UMAP**: Better than PCA for non-linear relationships
3. **KMeans**: Fast, interpretable, works well for unsupervised clustering


## Configuration

Edit these values in `train.py` to customize:

python
NUM_CLUSTERS = 5        # Number of clusters
IMG_SIZE = (128, 128)   # Image resolution
BATCH_SIZE = 4         # Batch size for processing
```


## Troubleshooting

### "Models not found" error
**Solution:** Run `python train.py` first

### "No images found"
**Solution:** Ensure `data/dataset/` contains image files (.jpg, .png, .jpeg)

### Image processing fails
**Solution:**
- Check image is not corrupted
- Try with different image format
- Ensure image file is readable

### Web interface won't start
**Solution:**
- Check port 5000 is not in use
- Install Flask: `pip install flask`
- Try different port in `app.py`: `app.run(debug=True, port=5001)`

### Out of memory
**Solution:**
- Reduce `BATCH_SIZE` in train.py
- Use fewer images for training
- Close other applications


## Performance Tips

1. **Better Clustering:**
   - Use 200+ animal images
   - Ensure images are similar quality
   - Try different `NUM_CLUSTERS` values

2. **Faster Training:**
   - Reduce image size
   - Increase BATCH_SIZE
   - Use GPU (with CUDA/cuDNN)

3. **Better Predictions:**
   - Use test images similar to training data
   - Ensure good lighting and clear animals


## Dataset Info

The project expects animal images in `data/dataset/`:
- **Formats**: JPG, PNG, JPEG
- **Resolution**: Any (will be resized to 224×224)
- **Count**: Minimum 10, recommended 100+
- **Content**: Animal photos


## Project Summary

| Stage | File | Purpose |
|-------|------|---------|
| 1 | `train.py` | Train clustering model, validate accuracy |
| 2 | `predict.py` | CLI predictions with error handling |
| 3 | `app.py` + `index.html` | Web interface with visualizations |


## Technologies Used

- **TensorFlow/Keras**: Deep learning & ResNet50
- **Scikit-learn**: KMeans clustering
- **UMAP**: Dimensionality reduction
- **Matplotlib**: Visualization
- **Flask**: Web framework
- **Plotly**: Interactive plots
- **Joblib**: Model serialization



# Unsupervised-Animal-Classification
