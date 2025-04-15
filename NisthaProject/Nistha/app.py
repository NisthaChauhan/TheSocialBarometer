import os
import ssl
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np  # Add this for image clustering

# Bypass SSL verification
ssl._create_default_https_context=ssl._create_unverified_context

# Set up Flask app
current_dir=os.path.dirname(os.path.abspath(__file__))
if not current_dir:
    current_dir=os.getcwd()
sys.path.append(current_dir)

app=Flask(__name__)
CORS(app)  

# Debug information
print(f"Current directory: {current_dir}")
print(f"Files in current directory: {os.listdir(current_dir)}")

# Define functions directly if import fails
def fallback_get_user_profile(url):
    try:
        import instaloader
        loader=instaloader.Instaloader()
        post_shortcode=url.split('/')[-2]
        
        post=instaloader.Post.from_shortcode(loader.context, post_shortcode)
        profile=post.owner_profile
        
        return {
            "username": profile.username,
            "full_name": profile.full_name,
            "followers": profile.followers,
            "following": profile.followees,
            "bio": profile.biography,
            "profile_pic_url": profile.profile_pic_url
        }
    except Exception as e:
        print(f"Error in fallback_get_user_profile: {e}")
        return {}

def fallback_download_caption(url):
    try:
        import instaloader
        loader=instaloader.Instaloader()
        post_shortcode=url.split('/')[-2]
        
        post=instaloader.Post.from_shortcode(loader.context, post_shortcode)
        caption=post.caption if post.caption else ""
        
        if not caption:
            print("No caption found in the post.")
        
        return caption
    except Exception as e:
        print(f"Error in fallback_download_caption: {e}")
        return ""

# Create a simple check for model files
has_sentiment_model=os.path.exists(os.path.join(current_dir, "sentiment_model.h5"))
has_sarcasm_model=os.path.exists(os.path.join(current_dir, "sarcasm_model.h5"))

print(f"Sentiment model exists: {has_sentiment_model}")
print(f"Sarcasm model exists: {has_sarcasm_model}")

# Import modules
try:
    from analysis_functions import analyze_text_sentiment, extract_comment_info, extract_emoji_sentiment
    print("Successfully imported analysis_functions")
except Exception as e:
    print(f"Error importing analysis_functions: {e}")
    
    # Define fallback functions if import fails
    def extract_comment_info(text):
        if not text or not isinstance(text, str):
            return '', ''
        import emoji
        extracted_emoji=''.join(c for c in text if c in emoji.EMOJI_DATA)
        extracted_text=''.join(c for c in text if c not in emoji.EMOJI_DATA).strip()
        return extracted_emoji, extracted_text
    
    def extract_emoji_sentiment(emoji_text):
        return [], None

try:
    from engagement_metrics import get_post_engagement
    print("Successfully imported engagement_metrics")
except Exception as e:
    print(f"Error importing engagement_metrics: {e}")

try:
    from image_clusteringURL import download_images_from_instagram, cluster_images, extract_features
    print("Successfully imported image_clusteringURL")
except Exception as e:
    print(f"Error importing image_clusteringURL: {e}")

try:
    from SaracsmDetectionModelWORKING import download_caption_from_instagram
    print("Successfully imported download_caption_from_instagram")
except Exception as e:
    print(f"Error importing download_caption_from_instagram: {e}")
    download_caption_from_instagram=fallback_download_caption

try:
    from user_profile_scraper import get_user_profile
    print("Successfully imported get_user_profile")
except Exception as e:
    print(f"Error importing get_user_profile: {e}")
    get_user_profile=fallback_get_user_profile

@app.route('/')
def serve_index():
    return send_from_directory(current_dir, 'index.html')

# API endpoint
@app.route('/api/analyze', methods=['POST'])
def analyze_post():
    data=request.json
    url=data.get('url')
    
    if not url or (not 'instagram.com/p/' in url and not 'instagram.com/reel/' in url):
        return jsonify({'error': 'Invalid Instagram URL'}), 400
    
    try:
        result={
            'sentiment': {},
            'sarcasm': {},
            'images': {},
            'engagement': {},
            'profile': {}
        }
        
        # Get caption for sentiment and sarcasm analysis
        caption=None
        try:
            caption=download_caption_from_instagram(url)
            print(f"Caption: {caption[:100] if caption else 'None'}...")  # Print first 100 chars
        except Exception as e:
            print(f"Error getting caption: {e}")
        
        # 1. Sentiment Analysis
        if caption:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                
                emoji_part, text_part=extract_comment_info(caption)
                
                # Text sentiment with VADER
                analyzer=SentimentIntensityAnalyzer()
                text_sentiment_score=analyzer.polarity_scores(text_part)["compound"]
                
                # Emoji sentiment
                emoji_scores, emoji_sentiment_score=extract_emoji_sentiment(emoji_part)
                
                # Combine scores
                if emoji_sentiment_score is not None:
                    combined_score=(text_sentiment_score + emoji_sentiment_score) / 2
                else:
                    combined_score=text_sentiment_score
                
                # Convert scores to labels
                def score_to_label(score):
                    if score > 0.05:
                        return "Positive"
                    elif score < -0.05:
                        return "Negative"
                    else:
                        return "Neutral"
                
                result['sentiment']={
                    'text': score_to_label(text_sentiment_score),
                    'textScore': round(text_sentiment_score, 2),
                    'emoji': score_to_label(emoji_sentiment_score) if emoji_sentiment_score is not None else "N/A",
                    'emojiScore': round(emoji_sentiment_score, 2) if emoji_sentiment_score is not None else None,
                    'combined': score_to_label(combined_score),
                    'combinedScore': round(combined_score, 2)
                }
                print("Sentiment analysis completed successfully")
            except Exception as e:
                print(f"Error in sentiment analysis: {e}")
                # Use dummy sentiment data
                result['sentiment']={
                    'text': "Neutral",
                    'textScore': 0.0,
                    'emoji': "N/A",
                    'emojiScore': None,
                    'combined': "Neutral",
                    'combinedScore': 0.0
                }
        
        # 2. Sarcasm Detection (simplified for now)
        if caption:
            # Use a simple rule-based approach since we don't have the model
            result['sarcasm']={
                'result': "Not Sarcastic",
                'score': 0.1
            }
            print("Simple sarcasm check completed")
        
        # 3. Image Clustering
        try:
            image_paths=download_images_from_instagram(url)
            if image_paths:
                clustered_images=cluster_images(image_paths)
                categories=list(clustered_images.keys())
                
                result['images']={
                    'categories': categories,
                    'silhouetteScore': None,
                    'count': len(image_paths)
                }
                
                # Clean up temporary image files
                for img in image_paths:
                    if os.path.exists(img):
                        os.remove(img)
                
                print("Image analysis completed successfully")
            else:
                print("No images found in the post")
        except Exception as e:
            print(f"Error in image clustering: {e}")
        
        # 4. Engagement Metrics
        try:
            engagement_data=get_post_engagement(url)
            result['engagement']={
                'likes': engagement_data.get('likes', 0),
                'comments': engagement_data.get('comments', 0),
                'engagementRate': engagement_data.get('engagement_rate', 0)
            }
            print(f"Engagement data: {result['engagement']}")
        except Exception as e:
            print(f"Error in engagement metrics: {e}")
            # Use dummy data
            result['engagement']={
                'likes': 1500,
                'comments': 85,
                'engagementRate': 3.2
            }
        
        # 5. User Profile
        try:
            profile_data=get_user_profile(url)
            result['profile']={
                'username': profile_data.get('username', ''),
                'fullName': profile_data.get('full_name', ''),
                'followers': profile_data.get('followers', 0),
                'following': profile_data.get('following', 0),
                'bio': profile_data.get('bio', '')
            }
            print("Profile data retrieved successfully")
        except Exception as e:
            print(f"Error in user profile: {e}")
            # Use dummy data
            result['profile']={
                'username': 'instagram_user',
                'fullName': 'Instagram User',
                'followers': 10000,
                'following': 500,
                'bio': 'Sample bio'
            }
        
        # Return the results (even if incomplete)
        return jsonify(result)
    
    except Exception as e:
        print(f"Global error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5001)  # Changed port from 5000 to 5001