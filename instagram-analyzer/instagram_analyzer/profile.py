import instaloader
import os
import requests
from datetime import datetime
from . import analysis_functions
'''
def download_image(url: str, output_dir: str, filename: str) -> str:
    """
    Download an image from a URL to the specified directory and return the relative path.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, filename)
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(full_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            relative_path = f"/static/uploads/images/{filename}"
            print(f"Image downloaded successfully to: {full_path}")
            return relative_path
        else:
            print(f"Failed to download image from {url}. Status code: {response.status_code}\n-----------------------------------------------------")
            return "/static/images/default-image.png"
    except Exception as e:
        print(f"Error downloading image from {url}: {str(e)}\n-----------------------------------------------------")
        return "/static/images/default-image.png"

def get_profile(url: str) -> dict:
    """
    Get profile information for the owner of an Instagram post, including locally stored profile picture.
    """
    try:
        print(f"Fetching profile for URL: {url}\n-----------------------------------------------------")
        loader = instaloader.Instaloader()
        
        # Extract shortcode from URL
        if '?' in url:
            url = url.split('?')[0]
        shortcode = url.split('/')[-2]
        print(f"Extracted shortcode: {shortcode}\n-----------------------------------------------------")
        
        # Get post and profile data
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        profile = post.owner_profile
        
        # Format numbers
        followers = profile.followers
        following = profile.followees
        posts = profile.mediacount
        
        # Format followers count
        followers_formatted = format_number(followers)
        following_formatted = format_number(following)
        posts_formatted = format_number(posts)
        
        # Download profile picture
        profile_pic_url = profile.profile_pic_url or ""
        profile_pic_path = "/static/images/default-profile.png"
        if profile_pic_url:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"profile_{shortcode}_{timestamp}.jpg"
            output_dir = os.path.join("WORKING"," instagram-analyzer","static", "uploads", "images")
            profile_pic_path = download_image(profile_pic_url, output_dir, filename)
        
        # Create result dictionary
        result = {
            "username": profile.username or "N/A",
            "full_name": profile.full_name or "N/A",
            "followers": followers,
            "followers_formatted": followers_formatted,
            "following": following,
            "following_formatted": following_formatted,
            "posts_count": posts,
            "posts_count_formatted": posts_formatted,
            "bio": profile.biography or "No bio available",
            "profile_pic": profile_pic_path,  # Use local path
            "is_private": profile.is_private,
            "is_verified": profile.is_verified
        }
        
        print("Profile data retrieved successfully:", result,"\n-----------------------------------------------------")
        return result
        
    except Exception as e:
        print(f"Error fetching user profile: {str(e)}\n-----------------------------------------------------")
        # Return a default profile with error indication
        return {
            "username": "N/A",
            "full_name": "N/A",
            "followers": 0,
            "followers_formatted": "0",
            "following": 0,
            "following_formatted": "0",
            "posts_count": 0,
            "posts_count_formatted": "0",
            "bio": "Profile unavailable",
            "profile_pic": "/static/images/default-profile.png",
            "is_private": False,
            "is_verified": False
        }

def format_number(num: int) -> str:
    """
    Format large numbers with K, M suffixes
    """
    if num >= 1000000:
        return f"{(num/1000000):.1f}M"
    elif num >= 1000:
        return f"{(num/1000):.1f}K"
    return str(num)
'''






def download_image(url: str, filename: str) -> str:
    """
    Download an image from a URL to the specified directory and return the relative path.
    """
    # Hardcoded output directory
    output_dir = r"C:\Nistha\Insta\WORKING\instagram-analyzer\static\uploads\images"
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, filename)
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(full_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            relative_path = f"/static/uploads/images/{filename}"
            print(f"Image downloaded successfully to: {full_path}")
            return relative_path
        else:
            print(f"Failed to download image from {url}. Status code: {response.status_code}\n-----------------------------------------------------")
            return "/static/images/default-image.png"
    except Exception as e:
        print(f"Error downloading image from {url}: {str(e)}\n-----------------------------------------------------")
        return "/static/images/default-image.png"

def get_profile(url: str) -> dict:
    """
    Get profile information for the owner of an Instagram post, including locally stored profile picture.
    """
    try:
        print(f"Fetching profile for URL: {url}\n-----------------------------------------------------")
        loader = instaloader.Instaloader()
        
        # Extract shortcode from URL
        if '?' in url:
            url = url.split('?')[0]
        shortcode = url.split('/')[-2]
        print(f"Extracted shortcode: {shortcode}\n-----------------------------------------------------")
        
        # Get post and profile data
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        profile = post.owner_profile
        
        # Format numbers
        followers = profile.followers
        following = profile.followees
        posts = profile.mediacount
        
        # Format followers count
        followers_formatted = format_number(followers)
        following_formatted = format_number(following)
        posts_formatted = format_number(posts)
        
        # Download profile picture
        profile_pic_url = profile.profile_pic_url or ""
        profile_pic_path = "/static/images/default-profile.png"
        if profile_pic_url:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"profile_{shortcode}_{timestamp}.jpg"
            # Using the download_image function with the hardcoded path
            profile_pic_path = download_image(profile_pic_url, filename)
        
        # Create result dictionary
        result = {
            "username": profile.username or "N/A",
            "full_name": profile.full_name or "N/A",
            "followers": followers,
            "followers_formatted": followers_formatted,
            "following": following,
            "following_formatted": following_formatted,
            "posts_count": posts,
            "posts_count_formatted": posts_formatted,
            "bio": profile.biography or "No bio available",
            "profile_pic": profile_pic_path,  # Use local path
            "is_private": profile.is_private,
            "is_verified": profile.is_verified
        }
        
        print("Profile data retrieved successfully:", result,"\n-----------------------------------------------------")
        return result
        
    except Exception as e:
        print(f"Error fetching user profile: {str(e)}\n-----------------------------------------------------")
        # Return a default profile with error indication
        return {
            "username": "N/A",
            "full_name": "N/A",
            "followers": 0,
            "followers_formatted": "0",
            "following": 0,
            "following_formatted": "0",
            "posts_count": 0,
            "posts_count_formatted": "0",
            "bio": "Profile unavailable",
            "profile_pic": "/static/images/default-profile.png",
            "is_private": False,
            "is_verified": False
        }

def format_number(num: int) -> str:
    """
    Format large numbers with K, M suffixes
    """
    if num >= 1000000:
        return f"{(num/1000000):.1f}M"
    elif num >= 1000:
        return f"{(num/1000):.1f}K"
    return str(num)











'''
def get_post_image_url(instagram_url: str) -> str:
    """
    Get post image URL and download it locally.
    """
    loader = instaloader.Instaloader(download_pictures=False, download_videos=False)
    
    # Extract shortcode from URL
    #if '?' in instagram_url:
    instagram_url = instagram_url.split('?')[0]
    shortcode = instagram_url.split('/')[-2]
    
    try:
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        img_url = post.url
        print("test123",img_url)
        # Download post image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"post_{shortcode}_{timestamp}.jpg"
        output_dir = os.path.join("static", "uploads", "images")
        relative_path = download_image(img_url, output_dir, filename)
        return relative_path
    except Exception as e:
        print(f"Failed to get post image: {e}\n-----------------------------------------------------")
        return "/static/images/default-post.png"

def get_profile_picture(instagram_url: str) -> str:
    """
    Get profile picture URL and download it locally.
    """
    loader = instaloader.Instaloader()
    
    try:
        shortcode = instagram_url.split('/')[-2]
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        profile_pic_url = post.owner_profile.profile_pic_url
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"profile_{shortcode}_{timestamp}.jpg"
        output_dir = os.path.join("static", "uploads", "images")
        relative_path = download_image(profile_pic_url, output_dir, filename)
        return relative_path
    except Exception as e:
        print(f"Failed to get profile picture: {e}\n-----------------------------------------------------")
        return "/static/images/default-profile.png"'''