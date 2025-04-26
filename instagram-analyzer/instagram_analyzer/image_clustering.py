import os
import instaloader
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from sklearn.metrics import silhouette_score
import analysis_functions

# Initialize ResNet models
model=None
labels_model=None

def initialize_models():
    """
    Initialize the ResNet models for feature extraction and classification
    """
    global model, labels_model
    
    if model is None:
        # Load pre-trained ResNet50 model for feature extraction
        model=ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    if labels_model is None:
        # Load pre-trained ResNet50 model for classification
        labels_model=ResNet50(weights='imagenet')

def get_image_category(img_path):
    """
    Get category of an image using ResNet50
    """
    initialize_models()
    
    try:
        img=image.load_img(img_path, target_size=(224, 224))
        img_array=image.img_to_array(img)
        img_array=np.expand_dims(img_array, axis=0)
        img_array=preprocess_input(img_array)
        preds=labels_model.predict(img_array)
        predictions=decode_predictions(preds, top=3)[0]
        
        # Return top 3 predictions
        return [
            {"label": pred[1].replace('_', ' ').title(), "score": float(pred[2])}
            for pred in predictions
        ]
    except Exception as e:
        print(f"Error getting image category: {e}")
        return [{"label": "Unknown", "score": 0.0}]

def download_images_from_instagram(url, output_dir):
    """
    Download images from an Instagram post
    """
    loader=instaloader.Instaloader()
    
    try:
        # Extract shortcode from URL
        if '?' in url:
            url=url.split('?')[0]
        shortcode=url.split('/')[-2]
        
        # Get post data
        post=instaloader.Post.from_shortcode(loader.context, shortcode)
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        images=[]
        
        # Handle single image or carousel
        if post.typename == 'GraphImage':
            img_url=post.url
            img_path=os.path.join(output_dir, f"{shortcode}_1.jpg")
            
            if not os.path.exists(img_path):
                analysis_functions.save_image_from_url(img_url, img_path)
                
            images.append(img_path)
            
        elif post.typename == 'GraphSidecar':
            for i, node in enumerate(post.get_sidecar_nodes()):
                img_url=node.display_url
                img_path=os.path.join(output_dir, f"{shortcode}_{i+1}.jpg")
                
                if not os.path.exists(img_path):
                    analysis_functions.save_image_from_url(img_url, img_path)
                    
                images.append(img_path)
                
        return images
    
    except Exception as e:
        print(f"Error downloading images: {e}")
        return []

def extract_features(img_path):
    """
    Extract features from an image using ResNet50
    """
    initialize_models()
    
    try:
        img=image.load_img(img_path, target_size=(224, 224))
        img_array=image.img_to_array(img)
        img_array=np.expand_dims(img_array, axis=0)
        img_array=preprocess_input(img_array)
        features=model.predict(img_array)
        return features[0]
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros(2048)  # Default feature vector size for ResNet50
def evaluate_clustering(features, clusters, n_clusters):
    """
    Evaluate clustering performance using multiple metrics
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    # Dictionary to store evaluation metrics
    metrics={}
    
    # Only calculate metrics if we have multiple clusters and enough samples
    if len(set(clusters)) > 1 and len(features) > n_clusters:
        # Silhouette Score: Higher is better (-1 to 1)
        metrics['silhouette_score']=float(silhouette_score(features, clusters))
        
        # Calinski-Harabasz Index: Higher is better
        metrics['calinski_harabasz_score']=float(calinski_harabasz_score(features, clusters))
        
        # Davies-Bouldin Index: Lower is better
        metrics['davies_bouldin_score']=float(davies_bouldin_score(features, clusters))
        
        # Calculate intra-cluster distances (cohesion)
        from sklearn.metrics import pairwise_distances
        import numpy as np
        
        # Calculate centroid for each cluster
        centroids=np.zeros((n_clusters, features.shape[1]))
        for i in range(n_clusters):
            cluster_points=features[clusters == i]
            if len(cluster_points) > 0:
                centroids[i]=np.mean(cluster_points, axis=0)
        
        # Calculate average distance to centroid for each cluster
        intra_cluster_distances={}
        for i in range(n_clusters):
            cluster_points=features[clusters == i]
            if len(cluster_points) > 0:
                distances=pairwise_distances(cluster_points, [centroids[i]])
                intra_cluster_distances[f'cluster_{i}_cohesion']=float(np.mean(distances))
        
        metrics['intra_cluster_distances']=intra_cluster_distances
        
        # Calculate inter-cluster distances (separation)
        if n_clusters > 1:
            inter_cluster_distances=pairwise_distances(centroids)
            metrics['inter_cluster_distances']=float(np.mean(inter_cluster_distances))
    else:
        metrics['silhouette_score']=None
        metrics['calinski_harabasz_score']=None
        metrics['davies_bouldin_score']=None
        metrics['intra_cluster_distances']={}
        metrics['inter_cluster_distances']=None
    
    return metrics
def cluster_images(image_paths):
    """
    Cluster images using KMeans
    """
    if len(image_paths) == 0:
        return {}
    
    if len(image_paths) == 1:
        # Get image category for single image
        categories=get_image_category(image_paths[0])
        main_category=categories[0]["label"]
        
        # Get relative path for frontend
        relative_path=image_paths[0].replace("\\", "/")
        if "static/" in relative_path:
            relative_path = "/" + relative_path.split("static/uploads/images", 1)[1]

            
        return {
            "clusters": {
                main_category: [{
                    "path": relative_path,
                    "categories": categories
                }]
            },
            "silhouette_score": None,
            "category_distribution": {main_category: 1}
        }
    
    # Extract features from all images
    features=[]
    for img_path in image_paths:
        feature=extract_features(img_path)
        features.append(feature)
    
    features=np.array(features)
    
    # Determine optimal number of clusters (up to 3)
    optimal_n_clusters=min(len(image_paths), 3)
    
    # Use KMeans to cluster images
    kmeans=KMeans(n_clusters=optimal_n_clusters, random_state=42)
    clusters=kmeans.fit_predict(features)
    
    # Calculate silhouette score if there are multiple clusters
    score=None
    if len(set(clusters)) > 1:
        score=float(silhouette_score(features, clusters))
    
    # Get categories for each image
    image_categories={}
    for i, img_path in enumerate(image_paths):
        categories=get_image_category(img_path)
        main_category=categories[0]["label"]
        
        if main_category not in image_categories:
            image_categories[main_category]=[]
        
        # Get relative path for frontend
        relative_path=img_path.replace("\\", "/")
        if relative_path.startswith("static/"):
            relative_path="/" + relative_path
            
        image_categories[main_category].append({
            "path": relative_path,
            "categories": categories,
            "cluster": int(clusters[i])
        })
    
    # Create category distribution
    category_distribution={k: len(v) for k, v in image_categories.items()}
    
    return {
        "clusters": image_categories,
        "silhouette_score": score,
        "category_distribution": category_distribution
    }

def analyze_images(url):
    """
    Main function to analyze images from an Instagram post
    """
    try:
        # Get absolute paths
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        static_dir = os.path.join(root_dir, 'static')
        output_dir = os.path.join(static_dir, 'uploads', 'images')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Image output directory: {output_dir}")  # Debug print
        
        # Download images
        image_paths = download_images_from_instagram(url, output_dir)
        
        if len(image_paths) == 0:
            return {
                "error": "No images found or post is a video",
                "clusters": {},
                "silhouette_score": None,
                "category_distribution": {},
                "image_count": 0
            }

        # Convert absolute paths to relative paths for frontend
        relative_image_paths = []
        for path in image_paths:
            # Get the path relative to the static directory
            rel_path = os.path.relpath(path, static_dir)
            # Convert Windows path separators to URL format and add /static/ prefix
            url_path = '/static/' + rel_path.replace('\\', '/')
            relative_image_paths.append(url_path)


        # Cluster images
        clustering_result = cluster_images(image_paths)
        
        # Update image paths in the result to use relative URLs
        if 'clusters' in clustering_result:
            for category, images in clustering_result["clusters"].items():
                for image in images:
                    if 'path' in image:
                        # Get the path relative to the static directory
                        rel_path = os.path.relpath(image['path'], static_dir)
                        # Convert Windows path separators to URL format and add /static/ prefix
                        image['path'] = '/static/' + rel_path.replace('\\', '/')
                        print(f"Converted image path: {image['path']}")  # Debug print
        
       
        # Add image count
        clustering_result["image_count"] = len(image_paths)
        
        return clustering_result
    
    except Exception as e:
        print(f"Error analyzing images: {e}")
        return {
            "error": str(e),
            "clusters": {},
            "silhouette_score": None,
            "category_distribution": {},
            "image_count": 0
        }
