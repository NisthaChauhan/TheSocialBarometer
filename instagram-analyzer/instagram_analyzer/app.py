'''from flask import Flask, render_template, request, jsonify, send_from_directory
import os
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
print(f"Root directory: {ROOT_DIR}\n-----------------------------------------------------")

# Configure paths
template_dir = os.path.join(ROOT_DIR, 'templates')
static_dir = os.path.join(ROOT_DIR, 'static')
upload_dir = os.path.join(static_dir, 'uploads', 'images')
images_dir = os.path.join(static_dir, 'images')

# Create necessary directories
os.makedirs(upload_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

app = Flask(__name__, 
           template_folder=template_dir,
           static_folder=static_dir,
           static_url_path='/static')

# Debug information
print(f"Template directory: {template_dir}\n-----------------------------------------------------")
print(f"Static directory: {static_dir}\n-----------------------------------------------------")
print(f"Upload directory: {upload_dir}\n-----------------------------------------------------")
print(f"Images directory: {images_dir}\n-----------------------------------------------------")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_post():
    """
    Main API endpoint to analyze an Instagram post
    """
    try:
        print("Received analyze request\n-----------------------------------------------------")
        data = request.get_json()
        print("Request data:", data,"\n-----------------------------------------------------")
        
        instagram_url = data.get('url', '')
        print("Instagram URL:", instagram_url,"\n-----------------------------------------------------")

        if not instagram_url or not ('instagram.com/p/' in instagram_url or 'instagram.com/reel/' in instagram_url):
            print("Invalid URL format\n-----------------------------------------------------")
            return jsonify({'error': 'Invalid Instagram URL'}), 400

        print(f"Analyzing post URL: {instagram_url}\n-----------------------------------------------------")
        
        # Run all analyses
        print("-----------------------------------------------------\nGetting profile data...\n-----------------------------------------------------")
        profile_data = get_profile(instagram_url)
        print("Profile data:", profile_data)
        
        print("-----------------------------------------------------\nGetting engagement data...\n-----------------------------------------------------")
        engagement_data = get_engagement(instagram_url)
        print("Engagement data:", engagement_data)
        
        print("-----------------------------------------------------\nGetting sentiment data...\n-----------------------------------------------------")
        sentiment_data = analyze_sentiment(instagram_url)
        print("Sentiment data:", sentiment_data)
        
        print("-----------------------------------------------------\nGetting sarcasm data...\n-----------------------------------------------------")
        sarcasm_data = detect_sarcasm(instagram_url)
        print("Sarcasm data:", sarcasm_data)
        
        print("-----------------------------------------------------\nGetting image data...\n-----------------------------------------------------")
        image_data = analyze_images(instagram_url)
        print("Image data:", image_data)

        # Combine all data
        result = {
            'profile': profile_data,
            'post': engagement_data,
            'sentiment': sentiment_data,
            'sarcasm': sarcasm_data,
            'images': image_data,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print("Sending response:", result,"\n-----------------------------------------------------")
        return jsonify(result)
    
    except Exception as e:
        print(f"Error analyzing post: {str(e)}\n-----------------------------------------------------")
        return jsonify({'error': str(e)}), 500

# Route to serve uploaded images
@app.route('/static/uploads/images/<path:filename>')
def serve_uploaded_image(filename):
    try:
        return send_from_directory(upload_dir, filename)
    except Exception as e:
        print(f"Error serving uploaded image: {str(e)}\n-----------------------------------------------------")
        return send_from_directory(images_dir, 'default-post.png')

# Route to serve static images
@app.route('/static/images/<path:filename>')
def serve_static_image(filename):
    try:
        return send_from_directory(images_dir, filename)
    except Exception as e:
        print(f"Error serving static image: {str(e)}\n-----------------------------------------------------")
        return jsonify({'error': 'Image not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)'''



from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import analysis modules
from instagram_analyzer.sentiment import analyze_sentiment
from instagram_analyzer.sarcasm import detect_sarcasm
from instagram_analyzer.image_clustering import analyze_images
from instagram_analyzer.engagement import get_engagement
from instagram_analyzer.profile import get_profile

# Hardcoded paths
ROOT_DIR = r"C:\Nistha\Insta\WORKING\instagram-analyzer"
print(f"Root directory: {ROOT_DIR}\n-----------------------------------------------------")

# Configure paths
template_dir = os.path.join(ROOT_DIR, 'templates')
static_dir = os.path.join(ROOT_DIR, 'static')
upload_dir = r"C:\Nistha\Insta\WORKING\instagram-analyzer\static\uploads\images"
images_dir = os.path.join(static_dir, 'images')

# Create necessary directories
os.makedirs(upload_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

app = Flask(__name__, 
           template_folder=template_dir,
           static_folder=static_dir,
           static_url_path='/static')

# Debug information
print(f"Template directory: {template_dir}\n-----------------------------------------------------")
print(f"Static directory: {static_dir}\n-----------------------------------------------------")
print(f"Upload directory: {upload_dir}\n-----------------------------------------------------")
print(f"Images directory: {images_dir}\n-----------------------------------------------------")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_post():
    """
    Main API endpoint to analyze an Instagram post
    """
    try:
        print("Received analyze request\n-----------------------------------------------------")
        data = request.get_json()
        print("Request data:", data,"\n-----------------------------------------------------")
        
        instagram_url = data.get('url', '')
        print("Instagram URL:", instagram_url,"\n-----------------------------------------------------")

        if not instagram_url or not ('instagram.com/p/' in instagram_url or 'instagram.com/reel/' in instagram_url):
            print("Invalid URL format\n-----------------------------------------------------")
            return jsonify({'error': 'Invalid Instagram URL'}), 400

        print(f"Analyzing post URL: {instagram_url}\n-----------------------------------------------------")
        
        # Run all analyses
        print("-----------------------------------------------------\nGetting profile data...\n-----------------------------------------------------")
        profile_data = get_profile(instagram_url)
        print("Profile data:", profile_data)
        
        print("-----------------------------------------------------\nGetting engagement data...\n-----------------------------------------------------")
        engagement_data = get_engagement(instagram_url)
        print("Engagement data:", engagement_data)
        
        print("-----------------------------------------------------\nGetting sentiment data...\n-----------------------------------------------------")
        sentiment_data = analyze_sentiment(instagram_url)
        print("Sentiment data:", sentiment_data)
        
        print("-----------------------------------------------------\nGetting sarcasm data...\n-----------------------------------------------------")
        sarcasm_data = detect_sarcasm(instagram_url)
        print("Sarcasm data:", sarcasm_data)
        
        print("-----------------------------------------------------\nGetting image data...\n-----------------------------------------------------")
        image_data = analyze_images(instagram_url)
        print("Image data:", image_data)

        # Combine all data
        result = {
            'profile': profile_data,
            'post': engagement_data,
            'sentiment': sentiment_data,
            'sarcasm': sarcasm_data,
            'images': image_data,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print("Sending response:", result,"\n-----------------------------------------------------")
        return jsonify(result)
    
    except Exception as e:
        print(f"Error analyzing post: {str(e)}\n-----------------------------------------------------")
        return jsonify({'error': str(e)}), 500

# Route to serve uploaded images
@app.route('/static/uploads/images/<path:filename>')
def serve_uploaded_image(filename):
    try:
        return send_from_directory(upload_dir, filename)
    except Exception as e:
        print(f"Error serving uploaded image: {str(e)}\n-----------------------------------------------------")
        return send_from_directory(images_dir, 'default-post.png')

# Route to serve static images
@app.route('/static/images/<path:filename>')
def serve_static_image(filename):
    try:
        return send_from_directory(images_dir, filename)
    except Exception as e:
        print(f"Error serving static image: {str(e)}\n-----------------------------------------------------")
        return jsonify({'error': 'Image not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)