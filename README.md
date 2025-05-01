```markdown
# 🌐 The Social Barometer

A real-time Instagram analytics dashboard that uses machine learning and natural language processing (NLP) to analyze user sentiment, detect sarcasm, and uncover visual content trends. Developed as my final year BSc Data Science capstone project.



## 📌 Overview

In today’s digital world, social media is more than just posts—it's emotion, influence, and engagement. The Social Barometer goes beyond likes and comments by extracting deeper insights from Instagram data using AI-powered tools.

From sentiment analysis to image clustering and sarcasm detection, this project is built to help brands, marketers, and creators understand what their audience really feels.



## 🎯 Features

- 📷 Instagram Data Scraper (via Instaloader)  
- 💬 Caption Sentiment Analysis using VADER & EmoSent  
- 😏 Sarcasm Detection with Deep Learning (98% Accuracy)  
- 🧠 Image Clustering using ResNet50 + KMeans (85.79% Accuracy)  
- 📊 Trend Visualization from hashtags, captions & images  
- ⏱️ Real-time Analysis Dashboard (Flask-based)



## 💡 Use Cases

- Campaign feedback tracking and engagement reporting  
- Brand sentiment monitoring during events or launches  
- Influencer/creator content performance analysis  
- Audience emotion mapping and trend prediction



## 🔧 Tech Stack

| Area           | Tools Used |
|----------------|------------|
| Programming    | Python 3.x |
| Backend        | Flask |
| Web Scraping   | Instaloader |
| NLP & Sentiment| VADER, EmoSent, NLTK |
| ML Models      | TensorFlow, Keras |
| Image Analysis | ResNet50 (Transfer Learning), KMeans |
| Data Handling  | Pandas, NumPy |
| Visualization  | Matplotlib, WordCloud |
| Others         | JSON, OS, datetime, PIL |


## 📈 Model Performance

| Module               | Accuracy |
|----------------------|----------|
| Image Clustering     | 85.79%   |
| Sentiment Analysis   | 58.67%   |
| Sarcasm Detection    | 98.00%   |


## 🔮 Future Scope

- 🌍 Multilingual post analysis using mBERT / XLM-R
- 🧠 Influence & network mapping with Graph Neural Networks
- 📊 Dashboards using Streamlit or Dash
- 🧭 Cross-platform support (Twitter/X, YouTube, Reddit, LinkedIn)
- 🗺️ Geotag and temporal trend modules


## 🧪 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/NisthaChauhan/TheSocialBarometer.git
   cd TheSocialBarometer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask app:
   ```bash
   python app.py
   ```

## 📄 Project Poster

🖼️ [View Poster (PDF)](./instagram-analyzer/instagram_analyzer/poster.png)  


## 📚 References

Includes tools and ideas from:
- [Instaloader](https://instaloader.github.io/)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [TensorFlow](https://www.tensorflow.org/)
- [EmoSent](https://pypi.org/project/emosent/)
- Research papers on sarcasm detection, sentiment classification, and image clustering


## 📬 Contact

Nistha Chauhan  
📧 chauhannistha.23@gmail.com
🔗 https://linkedin.com/in/nisthachauhan 
