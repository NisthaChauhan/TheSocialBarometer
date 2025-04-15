
import instaloader

def get_post_engagement(url):
    loader = instaloader.Instaloader()
    post_shortcode = url.split('/')[-2]
    
    try:
        post = instaloader.Post.from_shortcode(loader.context, post_shortcode)
        likes = post.likes
        comments = post.comments
        engagement_rate = (likes + comments) / post.owner_profile.followers if post.owner_profile.followers else 0
        
        return {
            "likes": likes,
            "comments": comments,
            "engagement_rate": round(engagement_rate * 100, 2)
        }
    except Exception as e:
        print(f"Error fetching engagement metrics: {e}")
        return {}
    
url=input("Enter the Instagram post URL: ")
print(get_post_engagement(url))
