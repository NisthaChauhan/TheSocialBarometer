# ''import os
# import instaloader
# from PIL import Image
# import requests
# from io import BytesIO
# import numpy as np
# from sklearn.cluster import KMeans
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.imagenet_utils import decode_predictions
# from sklearn.metrics import silhouette_score
# import analysis_functions
# from datetime import datetime

# # Initialize ResNet models
# model=None
# labels_model=None

# def initialize_models():
#     """
#     Initialize the ResNet models for feature extraction and classification
#     """
#     global model, labels_model
    
#     if model is None:
#         # Load pre-trained ResNet50 model for feature extraction
#         model=ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
#     if labels_model is None:
#         # Load pre-trained ResNet50 model for classification
#         labels_model=ResNet50(weights='imagenet')

# def get_image_category(img_path):
#     """
#     Get category of an image using ResNet50
#     """
#     initialize_models()
    
#     try:
#         img=image.load_img(img_path, target_size=(224, 224))
#         img_array=image.img_to_array(img)
#         img_array=np.expand_dims(img_array, axis=0)
#         img_array=preprocess_input(img_array)
#         preds=labels_model.predict(img_array)
#         predictions=decode_predictions(preds, top=3)[0]
        
#         # Return top 3 predictions
#         return [
#             {"label": pred[1].replace('_', ' ').title(), "score": float(pred[2])}
#             for pred in predictions
#         ]
#     except Exception as e:
#         print(f"Error getting image category: {e}")
#         return [{"label": "Unknown", "score": 0.0}]

# def download_images_from_instagram(url, saved_img_to_dir):
#     """
#     Download images from an Instagram post
#     """
#     try:
#         print(f"Attempting to download images from URL: {url}\n---------------------------------------------------------")
#         print(f"Output directory: {saved_img_to_dir}\n---------------------------------------------------------")
        
#         # Initialize Instaloader
#         loader = instaloader.Instaloader()
        
#         # Extract shortcode from URL
#         if '?' in url:
#             url = url.split('?')[0]
#         shortcode = url.split('/')[-2]
        
#         #print(f"Extracted shortcode: {shortcode}")
        
#         try:
#             # Get post data
#             post = instaloader.Post.from_shortcode(loader.context, shortcode)
#             #print(f"Successfully retrieved post. Type: {post.typename}")
            
#             # Create output directory if needed
#             os.makedirs(saved_img_to_dir, exist_ok=True)
            
#             images = []
            
#             # Handle single image or carousel
#             if post.typename == 'GraphImage':
#                 print("Processing single image post\n---------------------------------------------------------")
#                 img_url = post.url
#                 img_path = os.path.join(saved_img_to_dir, f"{shortcode}_1.jpg")
                
#                 if not os.path.exists(img_path):
#                     print(f"Downloading image to: {img_path}\n---------------------------------------------------------")
#                     analysis_functions.save_image_from_url(img_url, img_path)
#                     print("Single image saved successfully\n---------------------------------------------------------")
#                 else:
#                     print("Image already exists, skipping download\n---------------------------------------------------------")
#                 images.append(img_path)
                
#             elif post.typename == 'GraphSidecar':
#                 #print("Processing carousel post")
#                 for i, node in enumerate(post.get_sidecar_nodes()):
#                     img_url = node.display_url
#                     img_path = os.path.join(saved_img_to_dir, f"{shortcode}_{i+1}.jpg")
                    
#                     if not os.path.exists(img_path):
#                         print(f"Downloading image {i+1} to: {img_path}\n---------------------------------------------------------")
#                         analysis_functions.save_image_from_url(img_url, img_path)
#                         print(f"Image {i+1} saved successfully\n---------------------------------------------------------")
#                     else:
#                         print(f"Image {i+1} already exists, skipping download\n---------------------------------------------------------")
#                     images.append(img_path)
#             else:
#                 print(f"Unsupported post type: {post.typename}")
#                 return []
            
#             #print(f"Successfully processed {len(images)} images")
#             return images
            
#         except instaloader.exceptions.InstaloaderException as e:
#             print(f"Instaloader error: {str(e)}\n---------------------------------------------------------")
#             return []
            
#     except Exception as e:
#         print(f"Error in download_images_from_instagram: {str(e)}\n---------------------------------------------------------")
#         return []

# def extract_features(img_path):
#     """
#     Extract features from an image using ResNet50
#     """
#     initialize_models()
    
#     try:
#         img=image.load_img(img_path, target_size=(224, 224))
#         img_array=image.img_to_array(img)
#         img_array=np.expand_dims(img_array, axis=0)
#         img_array=preprocess_input(img_array)
#         features=model.predict(img_array)
#         print("Feature shape:", features.shape)
#         return features[0]
#     except Exception as e:
#         print(f"Error extracting features: {e}")
#         return np.zeros(2048)  # Default feature vector size for ResNet50

# def evaluate_clustering(features, clusters, n_clusters):
#     """
#     Evaluate clustering performance using multiple metrics
#     """
#     from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
#     # Dictionary to store evaluation metrics
#     metrics={}
    
#     # Only calculate metrics if we have multiple clusters and enough samples
#     if len(set(clusters)) > 1 and len(features) > n_clusters:
#         # Silhouette Score: Higher is better (-1 to 1)
#         metrics['silhouette_score']=float(silhouette_score(features, clusters))
        
#         # Calinski-Harabasz Index: Higher is better
#         metrics['calinski_harabasz_score']=float(calinski_harabasz_score(features, clusters))
        
#         # Davies-Bouldin Index: Lower is better
#         metrics['davies_bouldin_score']=float(davies_bouldin_score(features, clusters))
        
#         # Calculate intra-cluster distances (cohesion)
#         from sklearn.metrics import pairwise_distances
#         import numpy as np
        
#         # Calculate centroid for each cluster
#         centroids=np.zeros((n_clusters, features.shape[1]))
#         for i in range(n_clusters):
#             cluster_points=features[clusters == i]
#             if len(cluster_points) > 0:
#                 centroids[i]=np.mean(cluster_points, axis=0)
        
#         # Calculate average distance to centroid for each cluster
#         intra_cluster_distances={}
#         for i in range(n_clusters):
#             cluster_points=features[clusters == i]
#             if len(cluster_points) > 0:
#                 distances=pairwise_distances(cluster_points, [centroids[i]])
#                 intra_cluster_distances[f'cluster_{i}_cohesion']=float(np.mean(distances))
        
#         metrics['intra_cluster_distances']=intra_cluster_distances
        
#         # Calculate inter-cluster distances (separation)
#         if n_clusters > 1:
#             inter_cluster_distances=pairwise_distances(centroids)
#             metrics['inter_cluster_distances']=float(np.mean(inter_cluster_distances))
#     else:
#         metrics['silhouette_score']=None
#         metrics['calinski_harabasz_score']=None
#         metrics['davies_bouldin_score']=None
#         metrics['intra_cluster_distances']={}
#         metrics['inter_cluster_distances']=None
    
#     return metrics
# def cluster_images(image_paths):
#     """
#     Cluster images using KMeans
#     """
#     print("Clustering\n---------------------------------------------------------")


#     if len(image_paths) == 0:
#         return {}
    
#     if len(image_paths) == 1:
#         # Get image category for single image
#         categories=get_image_category(image_paths[0])
#         main_category=categories[0]["label"]
        
#         '''# Get relative path for frontend
#         relative_path=image_paths[0].replace("\\", "/")
#         if relative_path.startswith("instagram-analyzer/static/"):
#             relative_path="/" + relative_path
#         '''
#         relative_path=r"C:\Nistha\Insta\WORKING\instagram-analyzer\static"    
#         return {
#             "clusters": {
#                 main_category: [{
#                     "path": relative_path,
#                     "categories": categories
#                 }]
#             },
#             "silhouette_score": None,
#             "category_distribution": {main_category: 1}
#         }
    
#     # Extract features from all images
#     features=[]
#     for img_path in image_paths:
#         feature=extract_features(img_path)
#         features.append(feature)
    
#     features=np.array(features)
    
#     # Determine optimal number of clusters (up to 3)
#     optimal_n_clusters=min(len(image_paths), 3)
    
#     # Use KMeans to cluster images
#     kmeans=KMeans(n_clusters=optimal_n_clusters, random_state=42)
#     clusters=kmeans.fit_predict(features)
    
#     # Calculate silhouette score if there are multiple clusters
#     score=None
#     if len(set(clusters)) > 1:
#         score=float(silhouette_score(features, clusters))
    
#     # Get categories for each image
#     image_categories={}
#     for i, img_path in enumerate(image_paths):
#         categories=get_image_category(img_path)
#         main_category=categories[0]["label"]
        
#         if main_category not in image_categories:
#             image_categories[main_category]=[]
        
#         '''# Get relative path for frontend
#         relative_path = img_path.replace("\\", "/")'''
#         relative_path=r"C:\Nistha\Insta\WORKING\instagram-analyzer\static"
#         '''# Remove any parent directory references and ensure path starts with /static/
#         if "static" in relative_path:
#             parts = relative_path.split("static/")
#             relative_path = "/static/" + parts[-1]
#         else:
#             relative_path = "/static/" + relative_path'''
            
#         image_categories[main_category].append({
#             "path": relative_path,
#             "categories": categories,
#             "cluster": int(clusters[i])
#         })
    
#     # Create category distribution
#     category_distribution={k: len(v) for k, v in image_categories.items()}
    
#     return {
#         "clusters": image_categories,
#         "silhouette_score": score,
#         "category_distribution": category_distribution
#     }

# def analyze_images(post_url=None, direct_image_url=None):
#     """
#     Main function to analyze images from an Instagram post or direct image URL
#     """
#     try:
#         # Create output directory
#         output_dir = r"C:\Nistha\Insta\WORKING\instagram-analyzer\static"   #os.path.join("static", "uploads", "images")
#         os.makedirs(output_dir, exist_ok=True)
#         print(f"Created output directory: {output_dir}\n---------------------------------------------------------")

#         image_paths = []

#         # First try direct image URL if provided
#         if direct_image_url:
#             print(f"Attempting to download direct image from URL: {direct_image_url}")
#             try:
#                 # Generate a unique filename based on timestamp
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 filename = f"direct_image_{timestamp}.jpg"
#                 full_path = os.path.join(output_dir, filename)
#                 print(f"Full path for direct image: {full_path}\n---------------------------------------------------------")    	
#                 # Download the image
#                 response = requests.get(direct_image_url, stream=True)
#                 if response.status_code == 200:
#                     with open(full_path, 'wb') as f:
#                         for chunk in response.iter_content(chunk_size=8192):
#                             f.write(chunk)
#                     print(f"Successfully downloaded direct image to: {full_path}")
#                     # Store the frontend-accessible path
#                     frontend_path = f"/static/uploads/images/{filename}"
#                     print(f"Frontend path for direct image: {frontend_path}")
#                     image_paths.append(frontend_path)
#                 else:
#                     print(f"Failed to download direct image. Status code: {response.status_code}")
#             except Exception as e:
#                 print(f"Error downloading direct image: {str(e)}")

#         # If no direct image or direct download failed, try Instagram post
#         if not image_paths and post_url:
#             print(f"Attempting to download images from Instagram post: {post_url}")
#             try:
#                 # Initialize Instaloader
#                 loader = instaloader.Instaloader()
                
#                 # Extract shortcode from URL
#                 if '?' in post_url:
#                     post_url = post_url.split('?')[0]
#                 shortcode = post_url.split('/')[-2]
#                 print(f"Extracted shortcode: {shortcode}")
                
#                 # Get post data
#                 post = instaloader.Post.from_shortcode(loader.context, shortcode)
#                 print(f"Successfully retrieved post. Type: {post.typename}")
                
#                 # Handle single image or carousel
#                 if post.typename == 'GraphImage':
#                     print("Processing single image post")
#                     img_url = post.url
#                     filename = f"{shortcode}_1.jpg"
#                     full_path = os.path.join(output_dir, filename)
                    
#                     if not os.path.exists(full_path):
#                         print(f"Downloading image to: {full_path}")
#                         response = requests.get(img_url, stream=True)
#                         if response.status_code == 200:
#                             with open(full_path, 'wb') as f:
#                                 for chunk in response.iter_content(chunk_size=8192):
#                                     f.write(chunk)
#                             print("Single image saved successfully")
#                         else:
#                             print(f"Failed to download image. Status code: {response.status_code}")
#                     else:
#                         print("Image already exists, skipping download")
#                     # Store the frontend-accessible path
#                     frontend_path = f"/static/uploads/images/{filename}"
#                     print(f"Frontend path for single image: {frontend_path}")
#                     image_paths.append(frontend_path)
                    
#                 elif post.typename == 'GraphSidecar':
#                     print("Processing carousel post")
#                     for i, node in enumerate(post.get_sidecar_nodes()):
#                         img_url = node.display_url
#                         filename = f"{shortcode}_{i+1}.jpg"
#                         full_path = os.path.join(output_dir, filename)
                        
#                         if not os.path.exists(full_path):
#                             print(f"Downloading image {i+1} to: {full_path}")
#                             response = requests.get(img_url, stream=True)
#                             if response.status_code == 200:
#                                 with open(full_path, 'wb') as f:
#                                     for chunk in response.iter_content(chunk_size=8192):
#                                         f.write(chunk)
#                                 print(f"Image {i+1} saved successfully")
#                             else:
#                                 print(f"Failed to download image {i+1}. Status code: {response.status_code}")
#                         else:
#                             print(f"Image {i+1} already exists, skipping download")
#                         # Store the frontend-accessible path
#                         frontend_path = f"/static/uploads/images/{filename}"
#                         print(f"Frontend path for carousel image {i+1}: {frontend_path}")
#                         image_paths.append(frontend_path)
#                 else:
#                     print(f"Unsupported post type: {post.typename}")
                
#             except Exception as e:
#                 print(f"Error downloading from Instagram post: {str(e)}")

#         if not image_paths:
#             result = {
#                 "error": "No images found or could not download",
#                 "clusters": {},
#                 "silhouette_score": None,
#                 "category_distribution": {},
#                 "image_count": 0
#             }
#             print("No images available for analysis:", result)
#             return result

#         # Create a simple result structure with the image paths
#         result = {
#             "image_count": len(image_paths),
#             "images": image_paths,
#             "category_distribution": {"Post Images": len(image_paths)},
#             "clusters": {
#                 "Post Images": [{"path": path} for path in image_paths]
#             }
#         }

#         print("Final analysis result:", result)
#         return result

#     except Exception as e:
#         print(f"Error analyzing images: {str(e)}")
#         return {
#             "error": str(e),
#             "clusters": {},
#             "silhouette_score": None,
#             "category_distribution": {},
#             "image_count": 0
#         }
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
model = None
labels_model = None

def initialize_models():
    """
    Initialize the ResNet models for feature extraction and classification
    """
    global model, labels_model
    
    if model is None:
        # Load pre-trained ResNet50 model for feature extraction
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    if labels_model is None:
        # Load pre-trained ResNet50 model for classification
        labels_model = ResNet50(weights='imagenet')

def get_image_category(img_path):
    """
    Get category of an image using ResNet50
    """
    initialize_models()
    
    try:
        print("***********************************inside TRY\n***********************************")
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        preds = labels_model.predict(img_array)
        predictions = decode_predictions(preds, top=3)[0]
        # In get_image_category() function
        print(f"Categories for {img_path}: {predictions}")   
        # Return top 3 predictions
        return [
            {"label": pred[1].replace('_', ' ').title(), "score": float(pred[2])}
            for pred in predictions
        ]
    except Exception as e:
        print(f"Error getting image category: {e}\n-----------------------------------------------------")
        return [{"label": "Unknown", "score": 0.0}]

def download_images_from_instagram(url):
    """
    Download images from an Instagram post with hardcoded output directory
    """
    loader = instaloader.Instaloader()
    
    # Hardcoded output directory
    output_dir = r"C:\Nistha\Insta\WORKING\instagram-analyzer\static\uploads\images"
    
    try:
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract shortcode from URL
        if '?' in url:
            url = url.split('?')[0]
        shortcode = url.split('/')[-2]
        
        # Get post data
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        
        images = []
        
        # Handle single image or carousel
        if post.typename == 'GraphImage':
            img_url = post.url
            img_path = os.path.join(output_dir, f"{shortcode}_1.jpg")
            
            if not os.path.exists(img_path):
                analysis_functions.save_image_from_url(img_url, img_path)
                
            images.append(img_path)
            
        elif post.typename == 'GraphSidecar':
            for i, node in enumerate(post.get_sidecar_nodes()):
                img_url = node.display_url
                img_path = os.path.join(output_dir, f"{shortcode}_{i+1}.jpg")
                
                if not os.path.exists(img_path):
                    analysis_functions.save_image_from_url(img_url, img_path)
                    
                images.append(img_path)
                
        return images
    
    except Exception as e:
        print(f"Error downloading images: {e}\n-----------------------------------------------------")
        return []

def extract_features(img_path):
    """
    Extract features from an image using ResNet50
    """
    initialize_models()
    
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        return features[0]
    except Exception as e:
        print(f"Error extracting features: {e}\n-----------------------------------------------------")
        return np.zeros(2048)  # Default feature vector size for ResNet50

def evaluate_clustering(features, clusters, n_clusters):
    """
    Evaluate clustering performance using multiple metrics
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    # Dictionary to store evaluation metrics
    metrics = {}
    
    # Only calculate metrics if we have multiple clusters and enough samples
    if len(set(clusters)) > 1 and len(features) > n_clusters:
        # Silhouette Score: Higher is better (-1 to 1)
        metrics['silhouette_score'] = float(silhouette_score(features, clusters))
        
        # Calinski-Harabasz Index: Higher is better
        metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(features, clusters))
        
        # Davies-Bouldin Index: Lower is better
        metrics['davies_bouldin_score'] = float(davies_bouldin_score(features, clusters))
        
        # Calculate intra-cluster distances (cohesion)
        from sklearn.metrics import pairwise_distances
        import numpy as np
        
        # Calculate centroid for each cluster
        centroids = np.zeros((n_clusters, features.shape[1]))
        for i in range(n_clusters):
            cluster_points = features[clusters == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)
        
        # Calculate average distance to centroid for each cluster
        intra_cluster_distances = {}
        for i in range(n_clusters):
            cluster_points = features[clusters == i]
            if len(cluster_points) > 0:
                distances = pairwise_distances(cluster_points, [centroids[i]])
                intra_cluster_distances[f'cluster_{i}_cohesion'] = float(np.mean(distances))
        
        metrics['intra_cluster_distances'] = intra_cluster_distances
        
        # Calculate inter-cluster distances (separation)
        if n_clusters > 1:
            inter_cluster_distances = pairwise_distances(centroids)
            metrics['inter_cluster_distances'] = float(np.mean(inter_cluster_distances))
    else:
        metrics['silhouette_score'] = None
        metrics['calinski_harabasz_score'] = None
        metrics['davies_bouldin_score'] = None
        metrics['intra_cluster_distances'] = {}
        metrics['inter_cluster_distances'] = None
    
    return metrics

def cluster_images(image_paths):
    """
    Cluster images using KMeans
    """
    if len(image_paths) == 0:
        return {}
    
    if len(image_paths) == 1:
        # Get image category for single image
        categories = get_image_category(image_paths[0])
        main_category = categories[0]["label"]
        
        # Get relative path for frontend
        '''relative_path = image_paths[0].replace("\\", "/")
        if "/static/" not in relative_path:
            relative_path = "/static/uploads/images/" + os.path.basename(relative_path)
            '''
        # In cluster_images() function
        relative_path = img_path.replace(
            r"C:\Nistha\Insta\WORKING\instagram-analyzer",
            ""
        ).replace("\\", "/")
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
    features = []
    for img_path in image_paths:
        feature = extract_features(img_path)
        features.append(feature)
    
    features = np.array(features)
    
    # Determine optimal number of clusters (up to 3)
    optimal_n_clusters = min(len(image_paths), 3)
    
    # Use KMeans to cluster images
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    # Calculate silhouette score if there are multiple clusters
    score = None
    if len(set(clusters)) > 1:
        score = float(silhouette_score(features, clusters))
    
    # Get categories for each image
    image_categories = {}
    for i, img_path in enumerate(image_paths):
        categories = get_image_category(img_path)
        main_category = categories[0]["label"]
        
        if main_category not in image_categories:
            image_categories[main_category] = []
        
        # Get relative path for frontend
        relative_path = img_path.replace("\\", "/")
        if "/static/" not in relative_path:
            relative_path = "/static/uploads/images/" + os.path.basename(relative_path)
            
        image_categories[main_category].append({
            "path": relative_path,
            "categories": categories,
            "cluster": int(clusters[i])
        })
    
    # Create category distribution
    category_distribution = {k: len(v) for k, v in image_categories.items()}
    
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
        # Use hardcoded output directory
        output_dir = r"C:\Nistha\Insta\WORKING\instagram-analyzer\static\uploads\images"
        os.makedirs(output_dir, exist_ok=True)
        
        # Download images
        image_paths = download_images_from_instagram(url)
        
        if len(image_paths) == 0:
            result = {
                "error": "No images found or post is a video",
                "clusters": {},
                "silhouette_score": None,
                "category_distribution": {},
                "image_count": 0
            }
            print(result, "\n-----------------------------------------------------")
            return result
            
        # Cluster images
        clustering_result = cluster_images(image_paths)
        
        # Add image count
        clustering_result["image_count"] = len(image_paths)
        
        print(clustering_result, "\n-----------------------------------------------------")
        return clustering_result
    
    except Exception as e:
        print(f"Error analyzing images: {e}\n-----------------------------------------------------")
        result = {
            "error": str(e),
            "clusters": {},
            "silhouette_score": None,
            "category_distribution": {},
            "image_count": 0
        }
        print(result, "\n-----------------------------------------------------")
        return result