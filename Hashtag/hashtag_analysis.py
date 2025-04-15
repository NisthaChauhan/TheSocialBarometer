# hashtag_analysis.py
# Extended with: Word Cloud Visualization (external) & Hashtag Category Classification

import instaloader
import re
from collections import Counter, defaultdict
from visualization_functions import plot_hashtag_wordcloud  
# ----------------------------- CONFIG -----------------------------
CATEGORY_KEYWORDS = {
    'emotions': ['happy', 'love', 'fun', 'sad', 'vibes', 'mood'],
    'events': ['birthday', 'wedding', 'party', 'festival'],
    'brands': ['nike', 'adidas', 'zara', 'apple', 'samsung'],
    'places': ['india', 'travel', 'goa', 'paris', 'london', 'newyork'],
    'motivation': ['fitness', 'goals', 'grind', 'success', 'workhard'],
}

# --------------------------- FUNCTIONS ----------------------------
def get_captions(username, max_posts=100):
    # Create loader without login
    loader = instaloader.Instaloader(download_pictures=False,
                                    download_video_thumbnails=False,
                                    download_videos=False,
                                    save_metadata=False,
                                    download_comments=False)
    
    # Don't log in, just access the profile directly
    try:
        profile = instaloader.Profile.from_username(loader.context, username)
        captions = []
        print(f"Fetching posts from: {username}...")
        for i, post in enumerate(profile.get_posts()):
            if i >= max_posts:
                break
            if post.caption:
                captions.append(post.caption)
        return captions
    except instaloader.exceptions.ProfileNotExistsException:
        print(f"Profile {username} not found.")
        return []
    except instaloader.exceptions.LoginRequiredException:
        print(f"Instagram requires login to view this profile. The profile may be private.")
        return []
    except Exception as e:
        print(f"Error fetching posts: {e}")
        return []

def extract_hashtags(captions):
    hashtag_pattern = r"#\w+"
    hashtags = []
    for caption in captions:
        hashtags += re.findall(hashtag_pattern, caption.lower())
    return hashtags

def plot_top_hashtags(hashtags, top_n=10):
    import matplotlib.pyplot as plt
    counter = Counter(hashtags)
    most_common = counter.most_common(top_n)

    tags = [tag for tag, count in most_common]
    counts = [count for tag, count in most_common]

    plt.figure(figsize=(10, 6))
    plt.bar(tags, counts, color='skyblue')
    plt.title(f"Top {top_n} Hashtags")
    plt.xlabel("Hashtags")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def classify_hashtags(hashtags):
    category_map = defaultdict(list)
    for tag in hashtags:
        tag_clean = tag.strip("#")
        found = False
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in tag_clean for kw in keywords):
                category_map[category].append(tag)
                found = True
                break
        if not found:
            category_map['misc'].append(tag)
    
    print("\n🏷️ Hashtag Category Distribution:")
    for category, tags in category_map.items():
        print(f"- {category.capitalize()} ({len(tags)}): {', '.join(set(tags))[:100]}{'...' if len(tags) > 5 else ''}")
    
    return category_map

# --------------------------- MAIN RUNNER --------------------------

def run_hashtag_analysis(username):
    captions = get_captions(username)
    hashtags = extract_hashtags(captions)

    if not hashtags:
        print("No hashtags found.")
        return
    
    print(f"\n✅ Total hashtags found: {len(hashtags)}")
    plot_top_hashtags(hashtags)
    plot_hashtag_wordcloud(hashtags)         # ✅ Now uses your existing function
    classify_hashtags(hashtags)

if __name__ == "__main__":
    user = input("Enter Instagram username: ")
    run_hashtag_analysis(user)
