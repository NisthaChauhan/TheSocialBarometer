'''
ONLY CAPTION NEED TO CHANGE TO COMMENT

'''
import instaloader
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from SaracsmDetectionModelWORKING import *


def load_model_and_tokenizer(): 
    model = tf.keras.models.load_model("sarcasm_model.h5")
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    print("Model and tokenizer loaded!")
    return model, tokenizer

if __name__ == "__main__":
    print("Loading sarcasm detection model...")
    model, tokenizer = load_model_and_tokenizer()
    
    instagram_url = input("Enter the Instagram post URL: ")
    caption = download_caption_from_instagram(instagram_url)
    
    if caption:
        print("\nChecking caption for sarcasm...")
        result = detect_sarcasm(caption, model, tokenizer)
        print(f"Caption: {caption}\nSarcasm Detection: {result}\n")
    else:
        print("No caption found.")

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
'''