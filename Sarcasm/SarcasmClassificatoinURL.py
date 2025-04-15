'''
ONLY CAPTION 
NEED TO CHANGE TO COMMENT
'''
# analyze_instagram_caption.py

import instaloader
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Model parameters
max_length = 100
padding_type = 'post'

# Load the model and tokenizer
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("sarcasm_model.h5")
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    print("Model and tokenizer loaded!")
    return model, tokenizer


# Download Instagram caption
def download_caption(url):
    loader = instaloader.Instaloader()
    post_shortcode = url.split('/')[-2]
    try:
        post = instaloader.Post.from_shortcode(loader.context, post_shortcode)
        caption = post.caption if post.caption else ""
        return caption
    except Exception as e:
        print(f"Error downloading caption: {e}")
        return ""


# Run predictions
def predict_traits(caption, model, tokenizer):
    sequences = tokenizer.texts_to_sequences([caption])
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type)
    '''predictions = model.predict(padded)[0]
    
    traits = ["Sarcastic", "Ironic", "Humorous", "Exaggerated"]
    results = {trait: round(score * 100, 2) for trait, score in zip(traits, predictions)}
    '''
    prediction = model.predict(padded)[0][0] 
    sarcasm_percentage = round(prediction * 100, 2)
    results = {"Sarcastic": sarcasm_percentage}

    return results


# Main function to run the analysis
def main():
    print("Loading the multi-label model for caption analysis...")
    model, tokenizer = load_model_and_tokenizer()
    
    instagram_url = input("Enter the Instagram post URL: ")
    caption = download_caption(instagram_url)
    
    if caption:
        print(f"\nCaption: {caption}\n")
        results = predict_traits(caption, model, tokenizer)
        for trait, score in results.items():
            print(f"{trait}: {score}%")
    else:
        print("No caption found or unable to fetch the post.")


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

'''