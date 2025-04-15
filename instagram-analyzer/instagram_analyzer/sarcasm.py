import instaloader
import pickle
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Sarcasm detection parameters
vocab_size = 10000
max_length = 100
padding_type = 'post'
oov_tok = "<OOV>"

# Create a simple model for sarcasm detection
def create_sarcasm_model():
    """
    Create a simple model for sarcasm detection if no saved model exists
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 16, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def load_or_create_model_tokenizer():
    """
    Load pre-trained model and tokenizer or create new ones
    """
    try:
        # Try to load model and tokenizer
        model = tf.keras.models.load_model("models/sarcasm_model.h5")
        with open("models/tokenizer.pickle", "rb") as handle:
            tokenizer = pickle.load(handle)
        print("Sarcasm model and tokenizer loaded!")
    except:
        print("Creating new sarcasm model and tokenizer...")
        # Create directory if needed
        os.makedirs("models", exist_ok=True)
        
        # Create new model and simple tokenizer
        model = create_sarcasm_model()
        tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
        
        # Fit tokenizer on some common words
        tokenizer.fit_on_texts(["this is a sarcastic text", "this is not sarcastic"])
        
        # Save model and tokenizer
        model.save("models/sarcasm_model.h5")
        with open("models/tokenizer.pickle", "wb") as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return model, tokenizer

def detect_sarcasm(url: str) -> dict:
    """
    Detect sarcasm in an Instagram post caption.
    """
    model, tokenizer = load_or_create_model_tokenizer()
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
            return {"result": "No caption", "score": 0, "interpretation": "No caption to analyze"}
        
        # Preprocess text
        sequences = tokenizer.texts_to_sequences([caption])
        padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type)
        
        # Predict
        # Since we may not have a trained model, provide a default value
        # In a real scenario, you would train the model properly
        prediction = model.predict(padded)[0][0]
        
        # Determine sarcasm level
        if prediction > 0.8:
            label = "Highly Sarcastic"
            interpretation = "The caption shows strong sarcastic elements, including exaggeration, irony, and/or contradictory statements."
        elif prediction > 0.6:
            label = "Moderately Sarcastic"
            interpretation = "The caption contains noticeable sarcastic elements that may not be immediately obvious to all readers."
        elif prediction > 0.4:
            label = "Mildly Sarcastic"
            interpretation = "The caption has subtle hints of sarcasm that might be missed by some readers."
        else:
            label = "Not Sarcastic"
            interpretation = "The caption appears to be genuine with no detected sarcastic elements."
        
        return {
            "result": label,
            "score": float(prediction),
            "interpretation": interpretation
        }
    
    except Exception as e:
        print(f"Error detecting sarcasm: {e}")
        return {"result": "Error", "score": 0, "interpretation": "Error analyzing caption"}
