'''
DetectSentimentFromInstagram.py

Loads the trained sentiment model and tokenizer.
Fetches Instagram captions and analyzes text and emoji sentiment.
'''

import instaloader
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from analysis_functions import analyze_text_sentiment, extract_comment_info, extract_emoji_sentiment
import tensorflow as tf

# Sentiment analysis parameters
vocab_size = 10000
max_length = 100
padding_type = 'post'


def load_model_and_tokenizer():
    model = tf.keras.models.load_model("sentiment_model.h5")
    with open("sentiment_tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    print("Model and tokenizer loaded!")
    return model, tokenizer


def download_caption_from_instagram(url):
    loader = instaloader.Instaloader()
    post_shortcode = url.split('/')[-2]
    
    try:
        post = instaloader.Post.from_shortcode(loader.context, post_shortcode)
        caption = post.caption if post.caption else ""
        
        if not caption:
            print("No caption found in the post.")
        
        return caption
    except Exception as e:
        print(f"Error downloading caption: {e}")
        return ""


def detect_sentiment(caption, model, tokenizer):
    emoji_part, text_part = extract_comment_info(caption)
    
    # Text sentiment
    sequences = tokenizer.texts_to_sequences([text_part])
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type)
    text_sentiment_score = model.predict(padded)[0][0]
    
    # Emoji sentiment
    emoji_scores, emoji_sentiment_score = extract_emoji_sentiment(emoji_part)
    
    # Combine scores (equal weight)
    if emoji_sentiment_score is not None:
        combined_score = (text_sentiment_score + emoji_sentiment_score) / 2
    else:
        combined_score = text_sentiment_score
    
    # Convert scores to labels
    def score_to_label(score):
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    return {
        "text_sentiment": (score_to_label(text_sentiment_score), round(text_sentiment_score, 2)),
        "emoji_sentiment": (score_to_label(emoji_sentiment_score) if emoji_sentiment_score is not None else "N/A", round(emoji_sentiment_score, 2) if emoji_sentiment_score is not None else "N/A"),
        "combined_sentiment": (score_to_label(combined_score), round(combined_score, 2))
    }


def main():
    print("Loading sentiment analysis model...")
    model, tokenizer = load_model_and_tokenizer()
    
    instagram_url = input("Enter the Instagram post URL: ")
    caption = download_caption_from_instagram(instagram_url)
    
    if caption:
        print("\nAnalyzing caption sentiment...")
        results = detect_sentiment(caption, model, tokenizer)
        
        # Print the results
        print(f"Caption: {caption}\n")
        print(f"Text Sentiment: {results['text_sentiment'][0]} ({results['text_sentiment'][1]})")
        print(f"Emoji Sentiment: {results['emoji_sentiment'][0]} ({results['emoji_sentiment'][1]})")
        print(f"Overall Sentiment: {results['combined_sentiment'][0]} ({results['combined_sentiment'][1]})\n")
    else:
        print("No caption found.")


#for i in range(5):
if __name__ == "__main__":
    main()

'''
https://www.instagram.com/p/DGgpWPfzcUD/
https://www.instagram.com/p/DGbY6AtiZcI/
https://www.instagram.com/p/DGmHiybTcWe/?hl=en
https://www.instagram.com/p/DGjUHzMPWn9/?hl=en =>video
https://www.instagram.com/p/DGH0zZCpCJx/?hl=en => video
https://www.instagram.com/p/DGRhmgbSZ3A/?img_index=1
https://www.instagram.com/p/DGMQ3DIS1Jr/?hl=en&img_index=1
https://www.instagram.com/p/DGIEmQZJ3mR/?hl=en&img_index=1
https://www.instagram.com/p/CoBwYsNLpq6/
https://www.instagram.com/p/DGvA1cYxEgk/?img_index=1
https://www.instagram.com/p/DGqauqNCEVN/
https://www.instagram.com/p/DGfKllTqUpy/?img_index=1
https://www.instagram.com/p/DGsmg2CSdGs/?img_index=1
'''