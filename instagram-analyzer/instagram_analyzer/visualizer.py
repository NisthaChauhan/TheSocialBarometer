import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

class Visualizer:
    """Class for generating visualizations from text analysis results."""
    
    def __init__(self):
        """Initialize the Visualizer."""
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('english'))
    
    def plot_word_frequency(self, word_freq, top_n=20):
        """
        Plot a bar chart of word frequencies.
        
        Args:
            word_freq (dict): Dictionary with words as keys and frequencies as values
            top_n (int): Number of top words to include
        
        Returns:
            matplotlib.figure.Figure: The generated plot
        """
        # Sort and get top words
        sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        words, frequencies = zip(*sorted_freq)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(words, frequencies, color='skyblue')
        ax.set_title(f'Top {top_n} Word Frequencies')
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        
        return fig
    
    def generate_wordcloud(self, word_freq):
        """
        Generate a word cloud from word frequencies.
        
        Args:
            word_freq (dict): Dictionary with words as keys and frequencies as values
        
        Returns:
            WordCloud: The generated word cloud
        """
        wordcloud = WordCloud(width=800, height=400,
                             background_color='white',
                             max_words=100,
                             colormap='viridis').generate_from_frequencies(word_freq)
        return wordcloud
    
    def create_sentiment_wordclouds(self, df):
        """
        Creates word clouds for positive, negative, and neutral comments.
        
        Args:
            df (DataFrame): DataFrame containing comments and sentiment information
        """
        def combine_sentiments(row):
            if row.get('emoji_sentiment') == 'no_emoji':
                return row.get('text_sentiment')
            return row.get('emoji_sentiment') if row.get('emoji_sentiment') != 'neutral' else row.get('text_sentiment')
        
        if 'emoji_sentiment' in df.columns and 'text_sentiment' in df.columns:
            df['sentiment'] = df.apply(combine_sentiments, axis=1)
        elif 'sentiment' not in df.columns:
            print("Warning: No sentiment columns found in DataFrame")
            return

        sentiments = ['positive', 'negative', 'neutral']
        color_maps = {'positive': 'YlGn', 'negative': 'RdGy', 'neutral': 'Blues'}
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Word Clouds by Sentiment', fontsize=16, y=0.95)

        for ax, sentiment in zip(axes.flatten(), sentiments):
            sentiment_comments = df[df['sentiment'] == sentiment]['original_comment'] if 'original_comment' in df.columns else df[df['sentiment'] == sentiment]['text']
            text = " ".join(sentiment_comments.astype(str))
            
            word_tokens = word_tokenize(text.lower())
            filtered_text = " ".join(word for word in word_tokens 
                                  if word.isalnum() and word not in self.stop_words and len(word) > 2)

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
        return fig
    
    def plot_engagement_bar(self, likes, comments):
        """
        Plot a bar chart of engagement metrics.
        
        Args:
            likes (int): Number of likes
            comments (int): Number of comments
        
        Returns:
            matplotlib.figure.Figure: The generated plot
        """
        labels = ['Likes', 'Comments']
        values = [likes, comments]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, values, color=['#FF6F61', '#6B4226'])
        ax.set_title('Post Engagement')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Count')
        return fig

    def plot_sentiment_distribution(self, sentiment_counts):
        """
        Plot a pie chart of sentiment distribution.
        
        Args:
            sentiment_counts (dict): Dictionary with sentiment labels as keys and counts as values
        
        Returns:
            matplotlib.figure.Figure: The generated plot
        """
        labels = sentiment_counts.keys()
        sizes = sentiment_counts.values()

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#5DADE2', '#F5B041', '#EC7063'])
        ax.set_title('Sentiment Distribution')
        return fig
    
    def plot_hashtag_wordcloud(self, hashtags):
        """
        Generates a word cloud from a list of hashtags.
        
        Args:
            hashtags (list): List of hashtags
        
        Returns:
            matplotlib.figure.Figure: The generated plot
        """
        if not hashtags:
            print("No hashtags provided for word cloud.")
            return None

        text = ' '.join(hashtags)
        wc = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Hashtag Word Cloud")
        return fig
