import instaloader
import os
from . import analysis_functions

def get_profile(url: str) -> dict:
    """
    Get profile information for the owner of an Instagram post.
    """
    loader = instaloader.Instaloader()
    
    try:
        # Extract shortcode from URL
        if '?' in url:
            url = url.split('?')[0]
        shortcode = url.split('/')[-2]
        
        # Get post and profile data
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        profile = post.owner_profile
        
        # Save profile image
        profile_pic_filename = f"static/uploads/profile_{profile.username}.jpg"
        if not os.path.exists(profile_pic_filename):
            try:
                analysis_functions.save_image_from_url(profile.profile_pic_url, profile_pic_filename)
            except Exception as e:
                print(f"Error saving profile image: {e}")
                profile_pic_filename = None
        
        # Format data with metrics
        followers = profile.followers
        followees = profile.followees
        
        return {
            "username": profile.username,
            "full_name": profile.full_name,
            "followers": followers,
            "followers_formatted": analysis_functions.format_large_number(followers),
            "following": followees,
            "following_formatted": analysis_functions.format_large_number(followees),
            "bio": profile.biography,
            "profile_pic_url": f"/{profile_pic_filename}" if profile_pic_filename else None,
            "profile_url": f"https://www.instagram.com/{profile.username}/"
        }
    except Exception as e:
        print(f"Error fetching user profile: {e}")
        return {}
import instaloader

def get_post_image_url(instagram_url):
    loader = instaloader.Instaloader(download_pictures=False, download_videos=False)
    
    # No login — only public posts will work
    shortcode = instagram_url.split('/')[-2]
    
    try:
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        return post.url  # This is the direct image or video thumbnail URL
    except Exception as e:
        print(f"Failed to get post image: {e}")
        return ""

def get_profile_picture(instagram_url):
    loader = instaloader.Instaloader()
    
    try:
        shortcode = instagram_url.split('/')[-2]
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        return post.owner_profile.profile_pic_url
    except Exception as e:
        print(f"Failed to get profile picture: {e}")
        return ""

