from flask import Flask, render_template, request
import tweepy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import time
from datetime import datetime, timedelta

# Download VADER lexicon
nltk.download('vader_lexicon')

app = Flask(__name__)

import os

# Twitter API bearer token from environment variables
bearer_token = os.getenv('TWITTER_BEARER_TOKEN')

if not bearer_token:
    raise ValueError("TWITTER_BEARER_TOKEN environment variable is required")

# Initialize Twitter API v2 client
client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

# Sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Rate limiting tracking
last_request_time = None
request_count = 0
RATE_LIMIT_WINDOW = 15 * 60  # 15 minutes in seconds
MAX_REQUESTS_PER_WINDOW = 70  # Conservative limit to avoid hitting the ceiling

# Tweet cleaning function
def clean_text(text):
    return ' '.join(re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())

@app.route('/')
def index():
    return '''
        <h2>Twitter Sentiment Analysis</h2>
        <form action="/analyze" method="post">
            <input name="query" placeholder="Enter keyword or hashtag">
            <button type="submit">Analyze</button>
        </form>
    '''

@app.route('/analyze', methods=['POST'])
def analyze():
    global last_request_time, request_count
    
    query = request.form['query']
    
    # Rate limiting check
    current_time = datetime.now()
    if last_request_time is None:
        last_request_time = current_time
        request_count = 0
    
    # Reset count if 15 minutes have passed
    if (current_time - last_request_time).total_seconds() > RATE_LIMIT_WINDOW:
        request_count = 0
        last_request_time = current_time
    
    # Check if we've exceeded the rate limit
    if request_count >= MAX_REQUESTS_PER_WINDOW:
        time_to_wait = RATE_LIMIT_WINDOW - (current_time - last_request_time).total_seconds()
        return f"""
        <div style="padding: 20px; text-align: center; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px;">
            <h3>Rate limit exceeded</h3>
            <p>Please wait {int(time_to_wait/60)} minutes before making another request.</p>
            <p>Twitter allows limited API calls per 15-minute window.</p>
            <a href="/" style="margin-top: 10px; display: inline-block; padding: 8px 16px; background: #1da1f2; color: white; text-decoration: none; border-radius: 3px;">Back to Home</a>
        </div>
        """
    
    try:
        # Use a single, simple query to conserve API calls
        tweets = client.search_recent_tweets(
            query=query, 
            max_results=10,
            tweet_fields=['created_at']
        )
        request_count += 1
        last_request_time = current_time
        
        if not tweets or not tweets.data:
            return f"""
            <div style="padding: 20px; text-align: center;">
                <h3>No tweets found for "{query}"</h3>
                <p>Try these suggestions:</p>
                <ul style="text-align: left; display: inline-block;">
                    <li>Use more general terms (e.g., "AI" instead of "artificial intelligence")</li>
                    <li>Try popular hashtags (e.g., #python, #coding, #news)</li>
                    <li>Search for current trending topics</li>
                    <li>Remember: Twitter API only shows tweets from the last 7 days</li>
                </ul>
                <a href="/" style="margin-top: 20px; display: inline-block; padding: 10px 20px; background: #1da1f2; color: white; text-decoration: none; border-radius: 5px;">Try Another Search</a>
            </div>
            """
        
        results = []
        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        
        for tweet in tweets.data:
            text = clean_text(tweet.text)
            score = analyzer.polarity_scores(text)['compound']
            sentiment = 'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'
            sentiment_counts[sentiment] += 1
            
            # Color code the sentiment
            color = '#28a745' if sentiment == 'Positive' else '#dc3545' if sentiment == 'Negative' else '#6c757d'
            results.append(f'<div style="margin: 10px 0; padding: 10px; border-left: 4px solid {color};">{text} → <strong style="color: {color};">{sentiment}</strong> (score={score:.3f})</div>')

        summary = f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <h3>Analysis Results for "{query}"</h3>
            <p><strong>Tweets analyzed:</strong> {len(results)}</p>
            <p><strong>API calls remaining this window:</strong> {MAX_REQUESTS_PER_WINDOW - request_count}</p>
            <div style="display: flex; gap: 20px; margin: 10px 0;">
                <span style="color: #28a745;"><strong>Positive:</strong> {sentiment_counts['Positive']}</span>
                <span style="color: #dc3545;"><strong>Negative:</strong> {sentiment_counts['Negative']}</span>
                <span style="color: #6c757d;"><strong>Neutral:</strong> {sentiment_counts['Neutral']}</span>
            </div>
            <a href="/" style="margin-top: 10px; display: inline-block; padding: 8px 16px; background: #1da1f2; color: white; text-decoration: none; border-radius: 3px;">New Search</a>
        </div>
        """
        
        return summary + "".join(results)
    
    except tweepy.TooManyRequests:
        return """
        <div style="padding: 20px; text-align: center; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">
            <h3>Twitter API Rate Limit Exceeded</h3>
            <p>Please wait 15 minutes before trying again.</p>
            <a href="/" style="margin-top: 10px; display: inline-block; padding: 8px 16px; background: #1da1f2; color: white; text-decoration: none; border-radius: 3px;">Back to Home</a>
        </div>
        """
    except tweepy.Unauthorized:
        return """
        <div style="padding: 20px; text-align: center; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">
            <h3>Twitter API Authentication Failed</h3>
            <p>Please check your bearer token configuration.</p>
            <a href="/" style="margin-top: 10px; display: inline-block; padding: 8px 16px; background: #1da1f2; color: white; text-decoration: none; border-radius: 3px;">Back to Home</a>
        </div>
        """
    except Exception as e:
        return f"""
        <div style="padding: 20px; text-align: center; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">
            <h3>Error</h3>
            <p>Error fetching tweets: {str(e)}</p>
            <a href="/" style="margin-top: 10px; display: inline-block; padding: 8px 16px; background: #1da1f2; color: white; text-decoration: none; border-radius: 3px;">Back to Home</a>
        </div>
        """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
