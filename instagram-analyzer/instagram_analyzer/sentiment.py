import instaloader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from . import analysis_functions

def analyze_sentiment(url: str) -> dict:
    """
    Analyze sentiment of an Instagram post caption.
    """
    loader = instaloader.Instaloader()
    
    try:
        # Extract shortcode from URL
        if '?' in url:
            url = url.split('?')[0]
        shortcode = url.split('/')[-2]
        
        # Get post data
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        caption = post.caption if post.caption else ""
        
        if not caption:
            return {
                "text_sentiment": {"label": "N/A", "score": 0},
                "emoji_sentiment": {"label": "N/A", "score": 0},
                "combined_sentiment": {"label": "N/A", "score": 0},
                "emotional_tone": {"positivity": 0, "negativity": 0, "neutrality": 0, "emotionality": 0, "objectivity": 0}
            }
        
        # Separate text and emojis
        emoji_part, text_part = analysis_functions.extract_comment_info(caption)
        
        # Analyze text sentiment
        analyzer = SentimentIntensityAnalyzer()
        text_scores = analyzer.polarity_scores(text_part)
        text_sentiment_score = text_scores["compound"]
        
        # Determine text sentiment label
        if text_sentiment_score > 0.05:
            text_sentiment_label = "Positive"
        elif text_sentiment_score < -0.05:
            text_sentiment_label = "Negative"
        else:
            text_sentiment_label = "Neutral"
        
        # Analyze emoji sentiment
        emoji_scores, emoji_sentiment_score = analysis_functions.extract_emoji_sentiment(emoji_part)
        
        # Determine emoji sentiment label
        if emoji_sentiment_score is not None:
            if emoji_sentiment_score > 0.05:
                emoji_sentiment_label = "Positive"
            elif emoji_sentiment_score < -0.05:
                emoji_sentiment_label = "Negative"
            else:
                emoji_sentiment_label = "Neutral"
        else:
            emoji_sentiment_label = "N/A"
            emoji_sentiment_score = 0
        
        # Calculate combined sentiment
        if emoji_sentiment_label != "N/A":
            combined_score = (text_sentiment_score + emoji_sentiment_score) / 2
        else:
            combined_score = text_sentiment_score
        
        # Determine combined sentiment label
        if combined_score > 0.05:
            combined_sentiment_label = "Positive"
        elif combined_score < -0.05:
            combined_sentiment_label = "Negative"
        else:
            combined_sentiment_label = "Neutral"
        
        # Calculate emotional tone metrics
        positivity = max(0, min(1, (text_scores["pos"] + max(0, text_sentiment_score)) / 2))
        negativity = max(0, min(1, (text_scores["neg"] + max(0, -text_sentiment_score)) / 2))
        neutrality = max(0, min(1, text_scores["neu"]))
        emotionality = max(0, min(1, 1 - neutrality))
        objectivity = max(0, min(1, neutrality))
        
        return {
            "text_sentiment": {
                "label": text_sentiment_label,
                "score": round(text_sentiment_score, 2)
            },
            "emoji_sentiment": {
                "label": emoji_sentiment_label,
                "score": round(emoji_sentiment_score, 2) if emoji_sentiment_score is not None else 0
            },
            "combined_sentiment": {
                "label": combined_sentiment_label,
                "score": round(combined_score, 2)
            },
            "emotional_tone": {
                "positivity": round(positivity, 2),
                "negativity": round(negativity, 2),
                "neutrality": round(neutrality, 2),
                "emotionality": round(emotionality, 2),
                "objectivity": round(objectivity, 2)
            }
        }
    
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return {
            "text_sentiment": {"label": "Error", "score": 0},
            "emoji_sentiment": {"label": "Error", "score": 0},
            "combined_sentiment": {"label": "Error", "score": 0},
            "emotional_tone": {"positivity": 0, "negativity": 0, "neutrality": 0, "emotionality": 0, "objectivity": 0}
        }
