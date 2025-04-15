import instaloader


def get_post_engagement(url):
    loader = instaloader.Instaloader()
    shortcode = url.split('/')[-2]
    try:
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        data = {
                "shortcode": post.shortcode,
                "caption": post.caption,
                "hashtags": post.caption_hashtags,
                "mentions": post.caption_mentions,
                "accessibility_caption": post.accessibility_caption,
                "is_video": post.is_video,
                "comments_count": post.comments,
                "created_utc": str(post.date_utc),
                "created_local": str(post.date_local),
                "tagged_users": post.tagged_users,
                "thumbnail_url": post.url,
            }
        return data

    except Exception as e:
        print(f"Error fetching engagement metrics: {e}")
        return {}
    
url=input("Enter the Instagram post URL: ")
print(get_post_engagement(url))
