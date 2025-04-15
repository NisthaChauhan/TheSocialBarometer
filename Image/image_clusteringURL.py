# CNN (ResNet50) + KMeans

import os
import instaloader
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.imagenet_utils import decode_predictions 

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

# Load a pre-trained ResNet50 model for feature extraction
model=ResNet50(weights='imagenet',include_top=False,pooling='avg')

# Use ResNet50 categories
RESNET_LABELS_MODEL=ResNet50(weights='imagenet')


def get_image_category(img_path):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    img_array=np.expand_dims(img_array,axis=0)
    img_array=preprocess_input(img_array)
    preds=RESNET_LABELS_MODEL.predict(img_array)
    label=decode_predictions(preds,top=1)[0][0][1]
    return label


def download_images_from_instagram(url,max_images=3):
    loader=instaloader.Instaloader()
    post_shortcode=url.split('/')[-2]
    
    try:
        post=instaloader.Post.from_shortcode(loader.context,post_shortcode)
        images=[]
        
        # Handle single image or carousel
        if post.typename == 'GraphImage':
            img_url=post.url
            response=requests.get(img_url)
            img=Image.open(BytesIO(response.content))
            img.save("image_1.jpg")
            images.append("image_1.jpg")
        elif post.typename == 'GraphSidecar':
            for i,node in enumerate(post.get_sidecar_nodes()):
                if i >= max_images:
                    break
                img_url=node.display_url
                response=requests.get(img_url)
                img=Image.open(BytesIO(response.content))
                img.save(f"image_{i+1}.jpg")
                images.append(f"image_{i+1}.jpg")
        
        if not images:
            print("!!! No images found in the post!!!\n!!! Video posts are not supported!!!")
        
        return images
    except Exception as e:
        print(f"Error downloading images: {e}")
        return []


def extract_features(img_path):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    img_array=np.expand_dims(img_array,axis=0)
    img_array=preprocess_input(img_array)
    features=model.predict(img_array)
    return features[0]


from sklearn.metrics import silhouette_score  
'''
def cluster_images(image_paths):
    features=np.array([extract_features(img) for img in image_paths])
    
    # Use KMeans to cluster images
    n_clusters=len(image_paths)
    kmeans=KMeans(n_clusters=n_clusters,random_state=42)
    clusters=kmeans.fit_predict(features)
    if len(set(clusters)) > 1:
        score = silhouette_score(features, clusters)
        print(f"Silhouette Score: {score}")
    else:
        print("Not enough clusters for Silhouette Score.")


    clustered_images={}
    for i,img_path in enumerate(image_paths):
        cluster_label=clusters[i]
        category=get_image_category(img_path)
        if category not in clustered_images:
            clustered_images[category]=[]
        clustered_images[category].append(img_path)
    return clustered_images

'''
def cluster_images(image_paths):
    if len(image_paths) == 1:
        # If only 1 image, skip clustering
        category = get_image_category(image_paths[0])
        print(f"Single image detected. Category: {category}")
        return {category: image_paths}

    features = np.array([extract_features(img) for img in image_paths])

    # Use KMeans to cluster images
    n_clusters = min(len(image_paths), 2)  # Ensure at least 2 clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)

    if len(set(clusters)) > 1:
        score = silhouette_score(features, clusters)
        print(f"Silhouette Score: {score}")
    else:
        print("Not enough clusters for Silhouette Score.")

    clustered_images = {}
    for i, img_path in enumerate(image_paths):
        cluster_label = clusters[i]
        category = get_image_category(img_path)
        if category not in clustered_images:
            clustered_images[category] = []
        clustered_images[category].append(img_path)

    return clustered_images

def main():
    instagram_url=input("Enter the Instagram post URL: ")
    image_paths=download_images_from_instagram(instagram_url)
    
    if len(image_paths) == 0:
        print("No images to cluster. Exiting.")
        return
    clustered_images=cluster_images(image_paths)
    
    # Print the clustered results
    for category,images in clustered_images.items():
        print(f"\nCluster: {category}")
        for img_path in images:
            print(f"- {img_path}")


if __name__ == "__main__":
    main()


'''
https://www.instagram.com/p/DGgpWPfzcUD/
https://www.instagram.com/p/DGbY6AtiZcI/
https://www.instagram.com/p/DGmHiybTcWe/?hl=en
https://www.instagram.com/p/DGjUHzMPWn9/?hl=en =>video
https://www.instagram.com/p/DGH0zZCpCJx/?hl=en => video
https://www.instagram.com/p/DGRhmgbSZ3A/?img_index=1
https://www.instagram.com/p/DGMQ3DIS1Jr/?hl=en&img_index=1
https://www.instagram.com/p/DGIEmQZJ3mR/?hl=en&img_index=1
https://www.instagram.com/p/CoBwYsNLpq6/
https://www.instagram.com/p/DGvA1cYxEgk/?img_index=1
https://www.instagram.com/p/DGqauqNCEVN/
https://www.instagram.com/p/DGfKllTqUpy/?img_index=1
'''