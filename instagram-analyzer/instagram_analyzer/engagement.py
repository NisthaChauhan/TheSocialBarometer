import instaloader
import os
from . import analysis_functions
from datetime import datetime

def get_engagement(url: str) -> dict:
    """
    Get engagement metrics for an Instagram post.
    """
    try:
        # Extract shortcode from URL
        if '?' in url:
            url = url.split('?')[0]
        shortcode = url.split('/')[-2]
        
        # Initialize loader with quiet mode
        loader = instaloader.Instaloader(quiet=True)
        
        # Get post data
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        
        # Get image URL and download it
        image_url = None
        local_image_path = None
        try:
            if post.typename == 'GraphImage':
                image_url = post.url
            elif post.typename == 'GraphSidecar':
                # Get first image from carousel
                first_node = next(post.get_sidecar_nodes())
                image_url = first_node.display_url
            elif post.is_video:
                # For videos, use thumbnail
                image_url = post.url
                
            # Download the image if URL is available
            if image_url:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"post_{shortcode}_{timestamp}.jpg"
                output_dir = os.path.join("static", "uploads", "images")
                local_image_path = analysis_functions.save_image_from_url(image_url,  filename)
                
        except Exception as e:
            print(f"Error getting post image URL: {e}\n-----------------------------------------------------")
            local_image_path = "/static/images/default-post.png"
        
        # Calculate engagement rate
        profile = post.owner_profile
        followers = profile.followers
        likes = post.likes
        comments = post.comments
        
        engagement_rate = 0
        if followers > 0:
            engagement_rate = ((likes + comments) / followers) * 100
        
        return {
            "likes_count": likes,
            "comments_count": comments,
            "caption": post.caption if post.caption else "",
            "image_url": image_url,
            "local_image_path": local_image_path,
            "post_url": url,
            "engagement_rate": round(engagement_rate, 2),
            "post_date": post.date.strftime("%Y-%m-%d %H:%M:%S"),
            "is_video": post.is_video,
            "typename": post.typename
        }
    except Exception as e:
        print(f"Error fetching engagement metrics: {e}\n-----------------------------------------------------")
        return {
            "error": str(e),
            "likes_count": 0,
            "comments_count": 0,
            "caption": "",
            "image_url": None,
            "local_image_path": "/static/images/default-post.png",
            "post_url": url,
            "engagement_rate": 0,
            "post_date": None,
            "is_video": False,
            "typename": "Unknown"
        }
