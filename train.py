import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import joblib
import json

DATA_DIR = 'data/dataset/'
IMG_SIZE = (128, 128)
NUM_CLUSTERS = 5
BATCH_SIZE = 4

def load_images(data_dir, img_size):
    """Load all images from directory"""
    images = []
    filenames = []
    for file in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, file)
        if os.path.isfile(path) and file.lower().endswith(('.jpg', '.png', '.jpeg')):
            try:
                img = load_img(path, target_size=img_size)
                images.append(preprocess_input(img_to_array(img)))
                filenames.append(file)
            except Exception as e:
                print(f"Warning: Failed to load {file}: {e}")
    return np.array(images), filenames

print("=" * 60)
print("=" * 60)

print("\n1. Loading images...")
images, filenames = load_images(DATA_DIR, IMG_SIZE)
print(f"   ✓ Loaded {len(images)} images")

print("\n2. Extracting features using ResNet50...")
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
features = base_model.predict(images, batch_size=BATCH_SIZE, verbose=0)
print(f"   ✓ Features shape: {features.shape}")

print("\n3. Reducing dimensionality with UMAP...")
umap_model = umap.UMAP(n_neighbors=15, n_components=50, random_state=42)
features_umap = umap_model.fit_transform(features)
print(f"   ✓ UMAP features shape: {features_umap.shape}")

print("\n4. Clustering with KMeans (k={})...".format(NUM_CLUSTERS))
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
labels = kmeans.fit_predict(features_umap)
print(f"   ✓ Clustering complete")

silhouette = silhouette_score(features_umap, labels)
print(f"\n5. Model Metrics:")
print(f"   • Silhouette Score: {silhouette:.4f}")
print(f"   • Number of clusters: {NUM_CLUSTERS}")
print(f"   • Samples per cluster: {np.bincount(labels)}")

print("\n6. Saving models...")
joblib.dump(kmeans, 'kmeans_model.joblib')
joblib.dump(umap_model, 'umap_model.joblib')
joblib.dump(base_model, 'resnet_model.joblib')
print(f"   ✓ Models saved")

metadata = {
    'num_clusters': NUM_CLUSTERS,
    'silhouette_score': float(silhouette),
    'num_samples': len(images),
    'feature_dim': int(features.shape[1]),
    'umap_dim': int(features_umap.shape[1])
}
with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n7. Generating visualization...")
umap_2d = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
embedding_2d = umap_2d.fit_transform(features)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

scatter = axes[0].scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='tab20', s=50, alpha=0.6)
axes[0].set_title(f'Animal Clusters (K={NUM_CLUSTERS})', fontsize=14, fontweight='bold')
axes[0].set_xlabel('UMAP 1')
axes[0].set_ylabel('UMAP 2')
plt.colorbar(scatter, ax=axes[0], label='Cluster')

cluster_counts = np.bincount(labels)
axes[1].bar(range(NUM_CLUSTERS), cluster_counts, color='steelblue')
axes[1].set_title('Cluster Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel('Number of Samples')
axes[1].set_xticks(range(NUM_CLUSTERS))

plt.tight_layout()
plt.savefig('outputs/animal_clusters.png', dpi=100, bbox_inches='tight')
print(f"   ✓ Visualization saved to outputs/animal_clusters.png")

print("\n" + "=" * 60)
print("✓ TRAINING COMPLETE!")
print("=" * 60)
print(f"\nAccuracy Metric (Silhouette Score): {silhouette:.4f}")
print(f"Note: Silhouette score ranges from -1 to 1")
print(f"      > 0.5 = Good clustering, ✓ Model Quality: {'GOOD' if silhouette > 0.5 else 'FAIR'}")
