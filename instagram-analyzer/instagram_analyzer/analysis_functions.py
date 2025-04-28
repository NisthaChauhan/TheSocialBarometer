'''import regex as re
import emoji
from emoji import EMOJI_DATA
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Tuple, Optional, Any
import os
from io import BytesIO
import requests
from PIL import Image
import numpy as np
import time

def safe_request(url: str, max_retries: int = 3, retry_delay: int = 2) -> requests.Response:
    """
    Make a request with retry mechanism to handle rate limiting
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise e

def extract_comment_info(text: str) -> Tuple[str, str]:
    """
    Separates emojis from text in a comment.
    """
    if pd.isna(text) or not isinstance(text, str):      
        return '', ''
    extracted_emoji = ''.join(c for c in text if c in emoji.EMOJI_DATA)
    extracted_text = ''.join(c for c in text if c not in emoji.EMOJI_DATA).strip()
    return extracted_emoji, extracted_text

def analyze_text_sentiment(texts: List[str]) -> List[str]:
    """
    Analyzes text sentiment using VADER.
    """
    sentiments = []
    analyzer = SentimentIntensityAnalyzer()
    
    for text in texts:
        sentiment = analyzer.polarity_scores(str(text))["compound"]
        if sentiment > 0.05:
            sentiments.append('positive')
        elif sentiment < -0.05:
            sentiments.append('negative')
        else:
            sentiments.append('neutral')
    
    return sentiments

def get_sentiment_score(text: str) -> float:
    """
    Returns the sentiment score for a text.
    """
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(str(text))["compound"]

def extract_emoji_sentiment(emoji_text: str) -> Tuple[List, Optional[float]]:
    """
    Analyzes sentiment of emojis.
    """
    if not emoji_text:
        return [], None
    
    from emosent import get_emoji_sentiment_rank, get_emoji_sentiment_rank_multiple
    
    try:
        sentiment_data = get_emoji_sentiment_rank_multiple(emoji_text)
        scores = [item['emoji_sentiment_rank'] for item in sentiment_data]
        avg_score = sum(score['sentiment_score'] for score in scores) / len(scores) if scores else None
        return scores, avg_score
    except Exception as e:
        print(f"Error processing emojis {emoji_text}: {str(e)}")
        return [], None

def save_image_from_url(url: str, filename: str) -> str:
    """
    Downloads and saves an image from a URL to the local filesystem.
    """
    try:
        response = safe_request(url)
        img = Image.open(BytesIO(response.content))
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        img.save(filename)
        return filename
    except Exception as e:
        print(f"Error saving image from {url}: {str(e)}")
        return ""

def format_large_number(number: int) -> str:
    """
    Format large numbers for display (e.g., 1500 -> 1.5K)
    """
    if not isinstance(number, (int, float)):
        return str(number)
    
    if number >= 1_000_000:
        return f"{number/1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number/1_000:.1f}K"
    else:
        return str(number)

def calculate_engagement_rate(followers: int, likes: int, comments: int) -> float:
    """
    Calculate engagement rate based on followers, likes, and comments
    """
    if followers == 0:
        return 0
    else:
        return ((likes + comments) / followers) * 100
'''

import regex as re
import emoji
from emoji import EMOJI_DATA
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Tuple, Optional, Any
import os
from io import BytesIO
import requests
from PIL import Image
import numpy as np
import time

# Hardcoded path for image storage
IMAGE_STORAGE_PATH = r"C:\Nistha\Insta\WORKING\instagram-analyzer\static\uploads\images"

def safe_request(url: str, max_retries: int = 3, retry_delay: int = 2) -> requests.Response:
    """
    Make a request with retry mechanism to handle rate limiting
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise e

def extract_comment_info(text: str) -> Tuple[str, str]:
    """
    Separates emojis from text in a comment.
    """
    if pd.isna(text) or not isinstance(text, str):      
        return '', ''
    extracted_emoji = ''.join(c for c in text if c in emoji.EMOJI_DATA)
    extracted_text = ''.join(c for c in text if c not in emoji.EMOJI_DATA).strip()
    return extracted_emoji, extracted_text

def analyze_text_sentiment(texts: List[str]) -> List[str]:
    """
    Analyzes text sentiment using VADER.
    """
    sentiments = []
    analyzer = SentimentIntensityAnalyzer()
    
    for text in texts:
        sentiment = analyzer.polarity_scores(str(text))["compound"]
        if sentiment > 0.05:
            sentiments.append('positive')
        elif sentiment < -0.05:
            sentiments.append('negative')
        else:
            sentiments.append('neutral')
    
    return sentiments

def get_sentiment_score(text: str) -> float:
    """
    Returns the sentiment score for a text.
    """
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(str(text))["compound"]

def extract_emoji_sentiment(emoji_text: str) -> Tuple[List, Optional[float]]:
    """
    Analyzes sentiment of emojis.
    """
    if not emoji_text:
        return [], None
    
    from emosent import get_emoji_sentiment_rank, get_emoji_sentiment_rank_multiple
    
    try:
        sentiment_data = get_emoji_sentiment_rank_multiple(emoji_text)
        scores = [item['emoji_sentiment_rank'] for item in sentiment_data]
        avg_score = sum(score['sentiment_score'] for score in scores) / len(scores) if scores else None
        return scores, avg_score
    except Exception as e:
        print(f"Error processing emojis {emoji_text}: {str(e)}")
        return [], None

def save_image_from_url(url: str, filename: str) -> str:
    """
    Downloads and saves an image from a URL to the local filesystem using the hardcoded path.
    """
    try:
        response = safe_request(url)
        img = Image.open(BytesIO(response.content))
        
        # Use hardcoded path for saving images
        full_path = os.path.join(IMAGE_STORAGE_PATH, filename)
        
        # Create directory if needed
        os.makedirs(IMAGE_STORAGE_PATH, exist_ok=True)
        
        img.save(full_path)
        return f"/static/uploads/images/{filename}"  # Return web-accessible path
    except Exception as e:
        print(f"Error saving image from {url}: {str(e)}")
        return "/static/images/default-image.png"  # Return default image path on error

def format_large_number(number: int) -> str:
    """
    Format large numbers for display (e.g., 1500 -> 1.5K)
    """
    if not isinstance(number, (int, float)):
        return str(number)
    
    if number >= 1_000_000:
        return f"{number/1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number/1_000:.1f}K"
    else:
        return str(number)

def calculate_engagement_rate(followers: int, likes: int, comments: int) -> float:
    """
    Calculate engagement rate based on followers, likes, and comments
    """
    if followers == 0:
        return 0
    else:
        return ((likes + comments) / followers) * 100