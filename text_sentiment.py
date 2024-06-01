import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def analyze_text_sentiment(text):
    sia= SentimentIntensityAnalyzer()

    return sia.polarity_scores(text)

