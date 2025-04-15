import regex as re
import emoji
from emoji import EMOJI_DATA
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List
from emosent import get_emoji_sentiment_rank, get_emoji_sentiment_rank_multiple

def extract_comment_info(text):
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


def extract_emoji_sentiment(emoji_text):
    """
        Analyzes sentiment of emojis.2
    """
    if not emoji_text:
        return [], None
    try:
        sentiment_data = get_emoji_sentiment_rank_multiple(emoji_text)
        scores = [item['emoji_sentiment_rank'] for item in sentiment_data]
        avg_score = sum(score['sentiment_score'] for score in scores) / len(scores) if scores else None
        return scores, avg_score
    except Exception as e:
        print(f"Error processing emojis {emoji_text}: {str(e)}")
        return [], None
