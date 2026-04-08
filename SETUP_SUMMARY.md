# 3-Stage Unsupervised Animal Classification Project - Setup Summary

## ✅ What's Been Created

### Stage 1: Training (train.py) ✓
- **Features:**
  - Loads animal images from `data/dataset/`
  - Extracts features using pre-trained ResNet50
  - Reduces dimensionality with UMAP
  - Clusters using KMeans (5 clusters by default)
  - Calculates Silhouette Score for model quality
  - Generates 2D visualization
  - Saves all models to `.joblib` files
  - Exports metadata to `metadata.json`

### Stage 2: Prediction (predict.py) ✓
- **Features:**
  - Interactive CLI interface
  - Loads pre-trained KMeans, UMAP, ResNet50 models
  - Full input validation (file exists, is image format)
  - Error handling with user-friendly messages
  - Loop until user types 'quit'
  - Predicts cluster for new images

### Stage 3: Web Interface (app.py + templates/index.html) ✓
- **Features:**
  - Flask-based web dashboard
  - Model metrics display (Silhouette score, clusters, samples)
  - Architecture info explanation
  - Image upload form (click or drag & drop)
  - Real-time prediction API
  - Interactive 2D cluster visualization (Plotly)
  - Display 10 dataset sample images
  - Responsive design (desktop & mobile)
  - Color-coded results

### Supporting Files ✓
- **requirements.txt** - All Python dependencies
- **README.md** - Complete documentation
- **templates/index.html** - Web UI

## 🚀 How to Run

### Step 1: Train Model
```bash
python train.py
```
Outputs: 3 model files + visualization + metadata

### Step 2: Terminal Predictions
```bash
python predict.py
```
Interactive CLI for predictions

### Step 3: Web Interface
```bash
python app.py
```
Then open: http://localhost:5000

## 📊 Model Architecture
```
Image → ResNet50 (features) → UMAP (reduce dims) → KMeans (cluster)
```

## ✨ Key Improvements Made

1. **train.py:**
   - Added proper progress logging
   - Calculates metrics (Silhouette Score)
   - Better visualization (2 subplots)
   - Saves metadata for web interface
   - Error handling for corrupted images

2. **predict.py:**
   - Full input validation
   - Graceful error handling
   - Interactive loop instead of single prediction
   - Clear error messages
   - File existence checking

3. **app.py:**
   - Proper Flask structure with templates
   - API endpoints for predictions
   - Data visualization endpoint
   - Model caching for performance
   - File upload handling

4. **index.html:**
   - Modern, responsive UI
   - Beautiful gradient background
   - Interactive Plotly chart
   - Real-time prediction feedback
   - Sample image gallery
   - Mobile-friendly

## 📁 Project Structure
```
Unsupervised_Project/
├── train.py                    # Stage 1
├── predict.py                  # Stage 2
├── app.py                      # Stage 3
├── templates/
│   └── index.html             # Web UI
├── data/dataset/              # Animal images
├── outputs/                   # Visualizations
├── kmeans_model.joblib        #
├── umap_model.joblib          # Trained models
├── resnet_model.joblib        #
├── metadata.json              # Model info
├── requirements.txt           # Dependencies
└── README.md                  # Full documentation
```

## 🎯 How It Works

1. **Training Phase:**
   - Each animal image loaded and resized to 224×224
   - ResNet50 extracts 2048-dim features from each image
   - UMAP reduces to 50 dimensions (preserves structure)
   - KMeans groups images into 5 clusters based on features
   - Silhouette Score measures cluster quality
   - All processed, ready for inference

2. **Prediction Phase:**
   - New image loaded and preprocessed
   - Same feature extraction pipeline applied
   - Image assigned to nearest cluster
   - Result returned to user

3. **Web Interface:**
   - Displays model performance metrics
   - Allows image uploads for predictions
   - Shows all images in 2D space (UMAP)
   - Displays dataset samples

## 📈 Performance Metrics

- **Silhouette Score**: Measures cluster quality (-1 to 1)
  - > 0.5 = Good clustering
  - < 0.0 = Poor clustering
  - Use this instead of accuracy (it's unsupervised)

## ⚠️ Customization

Edit in `train.py`:
```python
NUM_CLUSTERS = 5        # Change number of clusters
IMG_SIZE = (224, 224)   # Change image resolution
BATCH_SIZE = 16         # Change batch size
```

## ✅ Status: READY TO USE

All 3 stages complete and integrated!
