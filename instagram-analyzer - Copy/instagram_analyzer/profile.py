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
        print("PROFILE:\n", profile)
        print("DICTIONARY\n", dir(profile))
        print("PROFILE username:\n", profile.username)
        print("PROFILE full name:\n", profile.full_name)
        print("PROFILE followers:\n", profile.followers)

        # Save profile image
        '''profile_pic_filename = f"static/uploads/profile_{profile.username}.jpg"
        try:
            analysis_functions.save_image_from_url(profile.profile_pic_url, profile_pic_filename)
        except Exception as e:
            print(f"Error saving profile image: {e}")
            profile_pic_filename = None
        '''
        
        
        '''result= {
            "username": profile.username,
            "full_name": profile.full_name,
            "followers": followers,
            "followers_formatted": analysis_functions.format_large_number(followers),
            "following": followees,
            "following_formatted": analysis_functions.format_large_number(followees),
            "bio": profile.biography,
            "profile_pic_url": profile.profile_pic_url,  # Return direct URL instead of local path
            "profile_pic": url,
        }'''
        result={
            "username": profile.username,
            "full_name": profile.full_name,
            "followers": profile.followers,
        }
        print("RESULT from profile :\n", result)
        return result
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

