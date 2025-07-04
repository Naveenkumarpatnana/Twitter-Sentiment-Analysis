from flask import Flask, render_template, request
import tweepy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

app = Flask(__name__)

# Twitter API keys (⚠️ don't expose in production!)
consumer_key = 'jyL7lcrei5O6gLPMy4EbxhfkX'
consumer_secret = 'XWXpoV3Uf0JVMzg3Z5mYsApoGIdErNPQOLtGZVgWMVSNhodxpR'
access_token = '1300808791961661442-rYwyuAr6L1qexeNCbpArppruYzrPXE'
access_token_secret = 'iBMPQ7wscqH66lfzY6apPeTjncvyJKUpOz7rf1sUJe8Rr'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

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
    query = request.form['query']
    tweets = api.search_tweets(q=query, lang='en', count=20)
    
    results = []
    for tweet in tweets:
        text = clean_text(tweet.text)
        score = analyzer.polarity_scores(text)['compound']
        sentiment = 'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'
        results.append(f"{text} → {sentiment} (score={score})")

    return "<br><br>".join(results)

if __name__ == '__main__':
    app.run(debug=True)
