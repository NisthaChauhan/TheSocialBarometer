import instaloader
import os
from . import analysis_functions

def get_engagement(url: str) -> dict:
    """
    Get engagement metrics for an Instagram post.
    """
    loader = instaloader.Instaloader()
    
    try:
        # Extract shortcode from URL
        if '?' in url:
            url = url.split('?')[0]
        shortcode = url.split('/')[-2]
        
        # Get post data
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        
        # Save post image to static folder
        image_filename = f"static/uploads/post_{shortcode}.jpg"
        if not os.path.exists(image_filename):
            try:
                if post.typename == 'GraphImage':
                    analysis_functions.save_image_from_url(post.url, image_filename)
                elif post.typename == 'GraphSidecar':
                    # Get first image from carousel
                    first_node = next(post.get_sidecar_nodes())
                    analysis_functions.save_image_from_url(first_node.display_url, image_filename)
                elif post.is_video:
                    # For videos, save thumbnail
                    analysis_functions.save_image_from_url(post.url, image_filename)
            except Exception as e:
                print(f"Error saving post image: {e}")
                image_filename = None
        
        # Format data
        data = {
            "shortcode": post.shortcode,
            "caption": post.caption,
            "hashtags": list(post.caption_hashtags),
            "mentions": list(post.caption_mentions),
            "accessibility_caption": post.accessibility_caption,
            "is_video": post.is_video,
            "comments_count": post.comments,
            "likes_count": post.likes,
            "created_utc": str(post.date_utc),
            "created_local": str(post.date_local),
            "tagged_users": list(post.tagged_users),
            "image_url": f"/{image_filename}" if image_filename else None,
            "post_url": url,
        }
        
        return data

    except Exception as e:
        print(f"Error fetching engagement metrics: {e}")
        return {}
