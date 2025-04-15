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
