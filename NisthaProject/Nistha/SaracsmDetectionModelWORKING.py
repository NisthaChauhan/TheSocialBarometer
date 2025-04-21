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
num_epochs = 5  # Reduced for faster training
training_size = 5000  # Reduced for faster training

# Load the dataset and train the model
file_path = "Sarcasm_Headlines_Dataset.json"  # Use relative path


def train_test_split(datastore):
    sentences, labels = [], []
    for _, row in datastore.iterrows():
        sentences.append(row['headline'])
        labels.append(row['is_sarcastic'])
    
    training_sentences = sentences[:training_size]
    testing_sentences = sentences[training_size:]   #data
    training_labels = labels[:training_size]
    testing_labels = labels[training_size:]     #labels

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type)

    return training_padded, np.array(training_labels), testing_padded, np.array(testing_labels), tokenizer


def sarcasm_detection_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
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
    return label, prediction


def save_model_and_tokenizer(model, tokenizer):
    model.save("sarcasm_model.h5")
    with open("tokenizer.pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Model and tokenizer saved!")


# Only run this if the script is executed directly
if __name__ == "__main__":
    print("Training sarcasm detection model...")
    
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found!")
            exit(1)
            
        # Load the dataset
        datastore = pd.read_json(file_path)
        print(f"Loaded {len(datastore)} records from the dataset")
        
        # Train the model
        training_padded, training_labels, testing_padded, testing_labels, tokenizer = train_test_split(datastore)
        model = sarcasm_detection_model()
        model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=1)
        test_loss,test_acc=model.evaluate(testing_padded,testing_labels)
        print(test_acc)

        # Save the model and tokenizer
        save_model_and_tokenizer(model, tokenizer)
        print("Training complete!")
    except Exception as e:
        print(f"Error during training: {e}")