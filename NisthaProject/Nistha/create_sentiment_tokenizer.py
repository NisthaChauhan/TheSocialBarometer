"""
Create a simple tokenizer for sentiment analysis.
This script will create a tokenizer that can be used by our sentiment analysis code.
"""

import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Sample texts to train the tokenizer
sample_texts = [
    "This is amazing! I love it so much!",
    "Terrible experience, would not recommend.",
    "It was okay, nothing special.",
    "The best thing I've ever seen in my life!",
    "Absolutely horrible, stay away from this.",
    "Average, not too bad but not great either.",
    "Just perfect! Couldn't ask for more!",
    "Disappointing, expected more for the price.",
    "Neutral feelings about this product."
]

# Create tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(sample_texts)

# Save the tokenizer
with open("sentiment_tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Sentiment tokenizer saved!")