# visualization_functions.py
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

def create_sentiment_wordclouds(df):
    """Creates word clouds for positive, negative, and neutral comments."""
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')
        stop_words = set(stopwords.words('english'))

    def combine_sentiments(row):
        if row['emoji_sentiment'] == 'no_emoji':
            return row['text_sentiment']
        return row['emoji_sentiment'] if row['emoji_sentiment'] != 'neutral' else row['text_sentiment']
    df['sentiment'] = df.apply(combine_sentiments, axis=1)

    sentiments = ['positive', 'negative', 'neutral']
    color_maps = {'positive': 'YlGn', 'negative': 'RdGy', 'neutral': 'Blues'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Word Clouds by Sentiment', fontsize=16, y=0.95)

    for ax, sentiment in zip(axes.flatten(), sentiments):
        sentiment_comments = df[df['sentiment'] == sentiment]['original_comment']
        text = " ".join(sentiment_comments.astype(str))
        
        word_tokens = word_tokenize(text.lower())
        filtered_text = " ".join(word for word in word_tokens 
                               if word.isalnum() and word not in stop_words and len(word) > 2)

        if filtered_text:
            wordcloud = WordCloud(width=800, height=400,
                                background_color='white',
                                max_words=50,
                                colormap=color_maps[sentiment]).generate(filtered_text)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'{sentiment.capitalize()} Comments (n={len(sentiment_comments)})')
        else:
            ax.text(0.5, 0.5, f'No {sentiment} comments found', ha='center', va='center')
        ax.axis('off')

    plt.tight_layout(pad=3.0)
    plt.show()

    print("\nSentiment Distribution:")
    print(df['sentiment'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

import matplotlib.pyplot as plt

def plot_engagement_bar(likes, comments):
    labels = ['Likes', 'Comments']
    values = [likes, comments]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=['#FF6F61', '#6B4226'])
    plt.title('Post Engagement')
    plt.xlabel('Metric')
    plt.ylabel('Count')
    plt.show()


def plot_sentiment_distribution(sentiment_counts):
    labels = sentiment_counts.keys()
    sizes = sentiment_counts.values()

    plt.figure(figsize=(7, 5))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#5DADE2', '#F5B041', '#EC7063'])
    plt.title('Sentiment Distribution')
    plt.show()
    
def plot_hashtag_wordcloud(hashtags):
    """
    Generates and displays a word cloud from a list of hashtags.
    """
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    if not hashtags:
        print("No hashtags provided for word cloud.")
        return

    text = ' '.join(hashtags)
    wc = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("Hashtag Word Cloud")
    plt.show()
