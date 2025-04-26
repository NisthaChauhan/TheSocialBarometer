from flask import Flask, render_template, request, jsonify
import os
import json
from werkzeug.utils import secure_filename
import pandas as pd
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import analysis modules
from instagram_analyzer.sentiment import analyze_sentiment
from instagram_analyzer.sarcasm import detect_sarcasm
from instagram_analyzer.image_clustering import analyze_images
from instagram_analyzer.engagement import get_engagement
from instagram_analyzer.profile import get_profile

# Set up absolute paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
template_dir = os.path.join(ROOT_DIR, 'templates')
static_dir = os.path.join(ROOT_DIR, 'static')

# Create upload directory if it doesn't exist
upload_dir = os.path.join(static_dir, 'uploads', 'images')
os.makedirs(upload_dir, exist_ok=True)

app = Flask(__name__, 
           template_folder=template_dir,
           static_folder=static_dir,
           static_url_path='/static')  # Explicitly set the static URL path

# Configure upload folder
app.config['UPLOAD_FOLDER'] = upload_dir
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Debug information
print(f"Root directory: {ROOT_DIR}")
print(f"Static directory: {static_dir}")
print(f"Upload directory: {app.config['UPLOAD_FOLDER']}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_post():
    """
    Main API endpoint to analyze an Instagram post
    """
    try:
        data=request.get_json()
        instagram_url=data.get('url', '')
        
        if not instagram_url or not ('instagram.com/p/' in instagram_url or 'instagram.com/reel/' in instagram_url):
            return jsonify({'error': 'Invalid Instagram URL'}), 400
        
        # Run all analyses
        profile_data=get_profile(instagram_url)
        engagement_data=get_engagement(instagram_url)
        sentiment_data=analyze_sentiment(instagram_url)
        sarcasm_data=detect_sarcasm(instagram_url)
        image_data=analyze_images(instagram_url)
        
        print("Engagement data\n\n",engagement_data)
        # Combine all data
        result={
            'profile': profile_data,
            'post': engagement_data,
            'sentiment': sentiment_data,
            'sarcasm': sarcasm_data,
            'images': image_data,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        
        print("RESULT:\n\n", result)
        return jsonify(result)
    
    except Exception as e:
        print(f"Error analyzing post: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)