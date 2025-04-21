import re
import nltk
import instaloader
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2

# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

### 1. Load and preprocess tweet emotion dataset ###
df = pd.read_csv(r"C:\Users\nisth\Downloads\tweet_emotions.csv\tweet_emotions.csv")
df["content"].fillna("unknown", inplace=True)

def process_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)

df["processed_Text"] = df["content"].apply(process_text)

encoder = LabelEncoder()
df["sentiment"] = encoder.fit_transform(df["sentiment"])
X = df["processed_Text"]
y = df["sentiment"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

### 2. Tokenizer ###
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)

train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

max_len = 100
train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding="post", truncating="post")
test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding="post", truncating="post")

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 50
num_classes = len(encoder.classes_)

### 3. Build and train model ###
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.002)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.002)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.002)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.002)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_sequences, y_train, epochs=50, batch_size=64, validation_data=(test_sequences, y_test))

### 4. Caption Emotion Analysis Function ###
def analyze_caption(url):
    loader = instaloader.Instaloader()
    if '?' in url:
        url = url.split('?')[0]
    shortcode = url.rstrip('/').split('/')[-1]

    try:
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        caption = post.caption or ""
        processed = process_text(caption)
        sequence = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
        prediction = model.predict(padded)
        predicted_emotion = encoder.inverse_transform([np.argmax(prediction)])[0]
        return {
            "caption": caption,
            "processed_caption": processed,
            "predicted_emotion": predicted_emotion
        }
    except Exception as e:
        return {"error": str(e)}

### Example usage ###
if __name__ == "__main__":
    result = analyze_caption("https://www.instagram.com/p/DGsmg2CSdGs/?img_index=1")
    print(result)
