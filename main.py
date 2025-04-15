# main.py
import pandas as pd
from analysis_functions import extract_comment_info, analyze_text_sentiment, extract_emoji_sentiment
from visualization_functions import create_sentiment_wordclouds
import emoji
from image_clustering import *      # setup_classifier,image_clustering,verify_path_structure,process_image,classify_cluster_images,analyze_cluster_predictions


def analyze_comments(comments):
    """
        Main function to analyze comments for both text and emoji sentiment
    """
    # Process comments
    comment_details = [extract_comment_info(comment) for comment in comments]
    emoji_list = [details[0] for details in comment_details]
    text_list = [details[1] for details in comment_details]
   
    # Create DataFrame
    df = pd.DataFrame({
        'original_comment': comments,
        'text': text_list,
        'emojis': emoji_list
    })

    # Analyze text sentiment
    df['text_sentiment'] = analyze_text_sentiment(df['text'])
    
    # Analyze emoji sentiment
    df['demojized_text'] = df['emojis'].apply(emoji.demojize)
    emoji_sentiments = df['emojis'].apply(extract_emoji_sentiment)
    df['emoji_sentiment_scores'] = emoji_sentiments.apply(lambda x: x[0])
    df['emoji_avg_score'] = emoji_sentiments.apply(lambda x: x[1])
    
    # Categorize emoji sentiment
    def categorize_sentiment(score):
        if pd.isna(score):
            return 'no_emoji'
        elif score > 0.05:
            return 'positive'
        elif score < -0.05:
            return 'negative'
        return 'neutral'
    
    df['emoji_sentiment'] = df['emoji_avg_score'].apply(categorize_sentiment)
    
    # Add combined sentiment
    def combine_sentiment(row):
        if row['emoji_sentiment'] == 'no_emoji':        # no emoji sentiment was detected.
            return row['text_sentiment']
        elif row['emoji_sentiment'] != 'neutral':       # emoji sentiment is either positive or negative.
            return row['emoji_sentiment']
        else:                                           # 'emoji_sentiment' is 'neutral'.   
            return row['text_sentiment']

    df['sentiment'] = df.apply(combine_sentiment, axis=1)
    return df

def post_segmentation():
    # Initialize model
    print("Loading ResNet50 model...")
    model = setup_classifier()
    
    # Set your base folder path containing cluster folders
    base_path = r"C:\Nistha\Insta\image\test"
    
    # Classify images in all clusters
    print("Starting classification...")
    results = classify_cluster_images(base_path, model)
    
    if not results:
        print("\nNo results to analyze. Please check the path and image files.")
        return
        
    # Analyze results
    print("\nAnalyzing clusters...")
    cluster_analysis = analyze_cluster_predictions(results)
    
    # Print detailed results
    print("\nClassification Results by Cluster:")
    for cluster_name, cluster_results in results.items():
        if not cluster_results:  # Skip empty clusters
            print(f"\nCluster '{cluster_name}': No valid images processed")
            continue
            
        print(f"\nCluster: {cluster_name}")
        print(f"Total images: {cluster_analysis[cluster_name]['total_images']}")
        print("Most common classes:")
        for class_name, count in cluster_analysis[cluster_name]['common_classes']:
            print(f"- {class_name}: {count} images")
        print(f"Unique classes found: {cluster_analysis[cluster_name]['unique_classes']}")
        
        print("\nDetailed image predictions:")
        for image_name, predictions in cluster_results.items():
            print(f"\n{image_name}:")
            for class_name, probability in predictions:
                print(f"- {class_name}: {probability:.2%}")

if __name__ == "__main__":
    file_path = "C:/Nistha/Insta/IG_Scrape500_csv.csv"
    try:
        df = pd.read_csv(file_path)
        result_df = analyze_comments(df['Biography'])       # Analyze comments
        create_sentiment_wordclouds(result_df)      #Create Wordcloud
        print("\nAnalysis completed successfully!")
        result_df=result_df['sentiment']
        result_df.to_csv("C:/Nistha/Insta/WORKING/final_sentiment.csv")
        print(post_segmentation())
    except Exception as e:
        print(f"Error: {str(e)}")
