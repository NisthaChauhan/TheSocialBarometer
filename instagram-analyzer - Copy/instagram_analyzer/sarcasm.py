import instaloader
import pickle
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Sarcasm detection parameters
vocab_size=10000
max_length=100
padding_type='post'
oov_tok="<OOV>"

# Create a simple model for sarcasm detection
def create_sarcasm_model():
    """
    Create a simple model for sarcasm detection if no saved model exists
    """
    '''model=tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 16, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])'''
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size+1,100,input_length=500),
        tf.keras.layers.Bidirectional( tf.keras.layers.LSTM(128)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_or_create_model_tokenizer():
    """
    Load pre-trained model and tokenizer or create new ones
    """
    '''try:
        # Try to load model and tokenizer
        model=tf.keras.models.load_model("models/sarcasm_model.keras")
        with open("models/tokenizer.pickle", "rb") as handle:
            tokenizer=pickle.load(handle)
        print("MY MODEL Sarcasm model and tokenizer loaded!")
        print(tokenizer)
    except:
        print("Creating new sarcasm model and tokenizer...")
        # Create directory if needed
        os.makedirs("models", exist_ok=True)
        
        # Create new model and simple tokenizer
        model=create_sarcasm_model()
        tokenizer=Tokenizer(num_words=vocab_size, oov_token=oov_tok)
        
        # Fit tokenizer on some common words
        tokenizer.fit_on_texts(["this is a sarcastic text", "this is not sarcastic"])
        
        # Save model and tokenizer
        model.save("models/sarcasm_model.h5")
        with open("models/tokenizer.pickle", "wb") as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)'''
    
    model_path = "C:/Nistha/Insta/WORKING/instagram-analyzer/models/sarcasm_model.h5"
    tokenizer_path = "C:/Nistha/Insta/WORKING/instagram-analyzer/models/tokenizer.pickle"
    
    try:
        # Try to load model and tokenizer
        model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)
        print(f"Sarcasm model and tokenizer loaded from {model_path}!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating new sarcasm model...")
        model = create_sarcasm_model()
        tokenizer = None  # You'll need to train or provide a tokenizer
        
    return model, tokenizer

def detect_sarcasm(url: str) -> dict:
    """
    Detect sarcasm in an Instagram post caption.
    """
    model, tokenizer=load_or_create_model_tokenizer()
    loader=instaloader.Instaloader()
    try:
        # Extract shortcode from URL
        if '?' in url:
            url=url.split('?')[0]
        shortcode=url.split('/')[-2]
        
        # Get post data
        post=instaloader.Post.from_shortcode(loader.context, shortcode)
        caption=post.caption if post.caption else ""
        
        if not caption:
            result= {"result": "No caption", "score": 0, "interpretation": "No caption to analyze"}
            print(result)
            
        # Preprocess text
        sequences=tokenizer.texts_to_sequences([caption])
        padded=pad_sequences(sequences, maxlen=max_length, padding=padding_type)
        prediction=model.predict(padded)[0][0]
        print("Reached prediction",prediction)
        # Determine sarcasm level
        if prediction > 0.9:
            label="Max-Level Sarcasm"
            interpretation="This caption is basically screaming sarcasm. It’s giving ✨drama✨, irony, and all the shade."
        elif prediction > 0.8:
            label="Super Sarcastic"
            interpretation="Yeah, this one’s *clearly* sarcastic. Like, it didn’t even try to hide it."
        elif prediction > 0.7:
            label="Pretty Sarcastic"
            interpretation="Strong sarcasm vibes here. Most people will catch the snark instantly."
        elif prediction > 0.6:
            label="Kinda Sarcastic"
            interpretation="You can tell it’s sarcastic, but it’s not going overboard. Just a lil sass."
        elif prediction > 0.5:
            label="Low-Key Sarcastic"
            interpretation="There's a sarcastic undertone here—like a side-eye in text form."
        elif prediction > 0.4:
            label="Casual Sarcasm"
            interpretation="Might be sarcasm, might just be passive-aggressive. Either way, it’s subtle."
        elif prediction > 0.3:
            label="Tiny Bit Sarcastic"
            interpretation="There's a lil sprinkle of sarcasm, but it’s barely there."
        elif prediction > 0.2:
            label="Microdose of Sarcasm"
            interpretation="Blink and you’ll miss it. Just a trace of sarcasm, if any."
        elif prediction > 0.1:
            label="Basically Genuine (But Who Knows)"
            interpretation="Seems real, but there’s a sliver of sass peeking through."
        else:
            label="Totally Genuine"
            interpretation="No sarcasm here—just vibes and sincerity. 🫶"

        print("Reached interpretation",label)
        result= {
            "caption":caption,
            "result": label,
            "score": float(prediction),
            "interpretation": interpretation
        }
        return result    
    except Exception as e:
        print(f"Error detecting sarcasm: {e}")
        result= {"result": "Error", "score": 0, "interpretation": "Error analyzing caption"}
        return result

