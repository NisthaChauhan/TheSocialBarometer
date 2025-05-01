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
import os


# Sarcasm detection parameters
vocab_size = 10000
embedding_dim = 16
max_length = 100
padding_type = 'post'
oov_tok = "<OOV>"
num_epochs = 30
training_size = 20000

# Load the dataset and train the model
file_path = r"C:\Nistha\Insta\WORKING\Sarcasm\Sarcasm_Headlines_Dataset.json"
datastore = pd.read_json(file_path)
'''
    PREPROCESSING
'''
import re
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

def augment_text(text):
    # Simple augmentation: randomly drop words
    words = text.split()
    if len(words) <= 3:
        return text
    drop_idx = np.random.randint(0, len(words))
    augmented = ' '.join(words[:drop_idx] + words[drop_idx+1:])
    return augmented



def train_test_split(datastore):
    sentences = [preprocess_text(row["headline"]) for _, row in datastore.iterrows()]
    labels = [row["is_sarcastic"] for _, row in datastore.iterrows()]
    
    training_sentences = sentences[:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[:training_size]
    testing_labels = labels[training_size:]
    
    augmented_sentences = [augment_text(sentence) for sentence in training_sentences]
    augmented_labels = training_labels.copy()

    training_sentences.extend(augmented_sentences)
    training_labels = np.concatenate([training_labels, augmented_labels])


    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type)

    return training_padded, np.array(training_labels), testing_padded, np.array(testing_labels), tokenizer


def sarcasm_detection_model():
    # embedding dimensions and add regularization
    embedding_dim = 128
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    return model


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


def detect_sarcasm(caption, model, tokenizer):
    sequences = tokenizer.texts_to_sequences([caption])
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type)
    
    prediction = model.predict(padded)[0][0]
    if prediction > 0.8:
            label = "Highly Sarcastic"
    elif prediction > 0.6:
        label = "Moderately Sarcastic"
    elif prediction > 0.4:
        label = "Mildly Sarcastic"
    else:
        label = "Not Sarcastic"    
    return label,prediction


def save_model_and_tokenizer(model, tokenizer):
    os.makedirs("C:/Nistha/Insta/WORKING/instagram-analyzer/models", exist_ok=True)
    
    # Save model and tokenizer in the same directory
    model.save("C:/Nistha/Insta/WORKING/instagram-analyzer/models/sarcasm_model.h5")
    with open("C:/Nistha/Insta/WORKING/instagram-analyzer/models/tokenizer.pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Model and tokenizer saved to C:/Nistha/Insta/WORKING/instagram-analyzer/models!")

if __name__ == "__main__":
    print("Training sarcasm detection model...")
    training_padded, training_labels, testing_padded, testing_labels, tokenizer = train_test_split(datastore)
    model = sarcasm_detection_model()
    model.fit(training_padded, 
              training_labels, 
              epochs=num_epochs, 
              validation_data=(testing_padded, testing_labels),
              callbacks=[tf.keras.callbacks.ProgbarLogger()], verbose=1)
    
    # Save the model and tokenizer
    save_model_and_tokenizer(model, tokenizer)
    print("Training complete!")
