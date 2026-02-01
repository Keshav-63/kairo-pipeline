# utils/sentiment_analyzer.py
from transformers import pipeline
from config.logger import logger

class SentimentAnalyzer:
    def __init__(self):
        try:
            self.sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            logger.info("SentimentAnalyzer initialized with 'cardiffnlp/twitter-roberta-base-sentiment-latest' model.")
        except Exception as e:
            logger.critical(f"Failed to initialize SentimentAnalyzer: {e}", exc_info=True)
            raise

    def get_sentiment(self, text: str) -> dict:
        """
        Analyzes the sentiment of the given text.
        Returns a dictionary with 'label' (Positive, Neutral, Negative) and 'score'.
        """
        if not text or not text.strip():
            return {"label": "Neutral", "score": 1.0}
        try:
            result = self.sentiment_pipeline(text)[0]
            
            # --- START OF FIX ---
            # The model now outputs lowercase labels. We need to standardize them.
            raw_label = result['label'].lower()
            
            # Map all possible labels to a standard, capitalized format
            label_map = {
                'positive': 'Positive',
                'label_2': 'Positive',
                'neutral': 'Neutral',
                'label_1': 'Neutral',
                'negative': 'Negative',
                'label_0': 'Negative'
            }
            # Use the mapped label, or default to "Neutral" if it's something unexpected
            result['label'] = label_map.get(raw_label, 'Neutral')
            # --- END OF FIX ---
            
            logger.debug(f"Sentiment for '{text[:50]}...': {result['label']} (score: {result['score']:.2f})")
            return result
        except Exception as e:
            logger.error(f"Sentiment analysis failed for text: '{text[:100]}...'. Error: {e}", exc_info=True)
            return {"label": "Error", "score": 0.0}