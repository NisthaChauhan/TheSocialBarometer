


# user_profile_scraper.py
import instaloader


def get_user_profile(url):
    loader = instaloader.Instaloader()
    post_shortcode = url.split('/')[-2]
    
    try:
        post = instaloader.Post.from_shortcode(loader.context, post_shortcode)
        profile = post.owner_profile
        
        return {
            "username": profile.username,
            "full_name": profile.full_name,
            "followers": profile.followers,
            "following": profile.followees,
            "bio": profile.biography,
            "profile_pic_url": profile.profile_pic_url
        }
    except Exception as e:
        print(f"Error fetching user profile: {e}")
        return {}
    
    
url=input("Enter the URL of the Instagram post: ")	
profile = get_user_profile(url)
print(profile)
