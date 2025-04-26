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
        '''print("POST username:\n",post.owner_username)
        print(post.url)
        print(post.likes)
        print(post.comments)
        print(post.caption)'''
        image_url = None
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
        except Exception as e:
            print(f"Error getting post image URL: {e}")
        
        return {
            "likes_count": post.likes,
            "comments_count": post.comments,
            "caption": post.caption if post.caption else "",
            "image_url": image_url,
            "post_url": url
        }
    except Exception as e:
        print(f"Error fetching engagement metrics: {e}")
        return {}
