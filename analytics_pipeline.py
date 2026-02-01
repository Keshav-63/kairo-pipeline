import asyncio
import yaml
from datetime import datetime, time, timezone
import pytz
from collections import Counter
import pytz
from config.logger import logger
from storage.mongodb_handler import AsyncMongoDBHandler
from utils.session_tracker import SessionTracker
from utils.sentiment_analyzer import SentimentAnalyzer
# Define the timezone for India
INDIA_TZ = pytz.timezone('Asia/Kolkata')
class SessionAnalytics:
    def __init__(self, user_id, self_speaker_label, sentiment_analyzer):
        self.user_id = user_id
        # This line is crucial. It takes the `self_speaker_label` passed in
        # and saves it as an attribute of the class instance (`self`).
        self.self_speaker_label = self_speaker_label 
        self.sentiment_analyzer = sentiment_analyzer
        self.total_speaking_time = 0
        self.total_listening_time = 0
        self.user_word_count = 0
        self.total_word_count = 0
        self.user_sentiments = []

    def process_segment(self, segment):
        duration = segment.get('end', 0) - segment.get('start', 0)
        word_count = len(segment.get('text', '').split())
        self.total_word_count += word_count

        # Here, we access the variable as `self.self_speaker_label`.
        # This works because it was defined in the __init__ method above.
        if segment.get('speaker') == self.self_speaker_label:
            self.total_speaking_time += duration
            self.user_word_count += word_count
            
            text_to_analyze = segment.get('text', '')
            if text_to_analyze:
                sentiment = self.sentiment_analyzer.get_sentiment(text_to_analyze)
                if sentiment and 'label' in sentiment:
                    self.user_sentiments.append(sentiment['label'])
        else:
            self.total_listening_time += duration
    
    # ... (rest of the SessionAnalytics class is the same)
    def get_metrics(self):
        if self.total_listening_time > 0:
            speaking_to_listening_ratio = self.total_speaking_time / self.total_listening_time
        else:
            speaking_to_listening_ratio = self.total_speaking_time

        sentiment_counts = Counter(self.user_sentiments)
        sentiment_metrics = {
            "positive_sentiment_count": sentiment_counts.get('Positive', 0),
            "neutral_sentiment_count": sentiment_counts.get('Neutral', 0),
            "negative_sentiment_count": sentiment_counts.get('Negative', 0),
        }

        metrics = {
            "totalSpeakingTime": self.total_speaking_time,
            "totalListeningTime": self.total_listening_time,
            "speakingToListeningRatio": speaking_to_listening_ratio,
            "socialTime": self.total_speaking_time + self.total_listening_time,
            "userWordCount": self.user_word_count,
            "totalWordCount": self.total_word_count,
        }
        metrics.update(sentiment_metrics)
        
        logger.info(f"Calculated Metrics: Speaking Time={metrics['totalSpeakingTime']:.2f}s, Listening Time={metrics['totalListeningTime']:.2f}s, Ratio={metrics['speakingToListeningRatio']:.2f}, User Words={metrics['userWordCount']}, Total Words={metrics['totalWordCount']}")
        logger.info(f"Sentiment Counts: Positive={sentiment_metrics['positive_sentiment_count']}, Neutral={sentiment_metrics['neutral_sentiment_count']}, Negative={sentiment_metrics['negative_sentiment_count']}")
        
        return metrics


class AnalyticsPipeline:
    def __init__(self, config):
        self.mongo = AsyncMongoDBHandler(config['mongodb']['uri'])
        self.tracker = SessionTracker(self.mongo, None)
        self.sentiment_analyzer = SentimentAnalyzer()

    async def process_session_for_analytics(self, session_id, user_id):
        logger.info(f"--- [ANALYTICS START] Processing session: {session_id} for user: {user_id} ---")
        try:
            self_speaker_label = await self.mongo.get_self_speaker_label(user_id)
            if not self_speaker_label:
                logger.warning(f"Could not identify 'self' speaker label for user {user_id}. Skipping analytics for session {session_id}.")
                await self.mongo.update_session_analytics_status(session_id, user_id, 'failed', {'error': 'Self speaker label not found.'})
                return

            logger.info(f"Identified user's speaker label as: '{self_speaker_label}'")
            # The 'self_speaker_label' variable is passed here
            analytics = SessionAnalytics(user_id, self_speaker_label, self.sentiment_analyzer)
            
            segments = await self.mongo.get_segments_for_session(session_id)
            if not segments:
                logger.warning(f"No transcript segments found for session {session_id}. Cannot process analytics.")
                await self.mongo.update_session_analytics_status(session_id, user_id, 'complete')
                return

            logger.info(f"Retrieved {len(segments)} transcript segments for processing.")
            
            for segment in segments:
                analytics.process_segment(segment)

            metrics = analytics.get_metrics()
            
            # Update the session date handling
            session_date = None
            if segments and 'createdAt' in segments[0]:
                created_at = segments[0]['createdAt']
                
                # Convert naive datetime to UTC if needed
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                
                # Convert to Indian timezone
                local_date = created_at.astimezone(INDIA_TZ).date()
                session_date = local_date
            else:
                session_doc = await self.mongo.sessions.find_one({"sessionId": session_id})
                updated_at = session_doc.get('updatedAt', datetime.utcnow())
                
                # Convert naive datetime to UTC if needed
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
                
                # Convert to Indian timezone
                session_date = updated_at.astimezone(INDIA_TZ).date()

            logger.info(f"Using session date (India timezone): {session_date}")
            await self.mongo.upsert_daily_analytics(user_id, session_date, metrics)
            # ...existing code...

            await self.mongo.update_session_analytics_status(session_id, user_id, status='complete')
            logger.info(f"--- [ANALYTICS SUCCESS] Successfully processed session: {session_id} ---")

        except Exception as e:
            logger.error(f"--- [ANALYTICS FAILED] Processing failed for {session_id}: {e} ---", exc_info=True)
            await self.mongo.update_session_analytics_status(session_id, user_id, 'failed', {'error': str(e)})

    async def run_pipeline(self):
        logger.info("Starting Kairo Plus Analytics Pipeline Polling Loop...")
        while True:
            try:
                global config
                user_ids = await self.mongo.get_distinct_users()
                if not user_ids:
                    logger.info("No users found in database. Sleeping.")
                else:
                    logger.info(f"Found {len(user_ids)} distinct users to check.")
                
                total_sessions_processed = 0
                for user_id in user_ids:
                    logger.info(f"Checking for unprocessed sessions for user: {user_id}")
                    unprocessed_sessions = await self.mongo.get_unprocessed_sessions_for_analytics(user_id)
                    
                    if not unprocessed_sessions:
                        logger.info(f"No new sessions to analyze for user {user_id}")
                        continue
                    
                    logger.info(f"Found {len(unprocessed_sessions)} new sessions to analyze for user {user_id}")
                    for session in unprocessed_sessions:
                        await self.process_session_for_analytics(session['sessionId'], user_id)
                        total_sessions_processed += 1

                logger.info(f"Polling loop finished. Processed {total_sessions_processed} sessions in this cycle.")
                
                interval = config.get('analytics_polling_interval', 300) 
                logger.info(f"Sleeping for {interval} seconds...")
                await asyncio.sleep(interval)
            except Exception as e:
                logger.critical(f"A critical error occurred in the analytics pipeline loop: {e}", exc_info=True)
                await asyncio.sleep(config.get('critical_error_sleep_seconds', 60))

async def main():
    global config
    with open('config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Initializing AnalyticsPipeline")
    pipeline = AnalyticsPipeline(config)
    logger.info("AnalyticsPipeline initialized, starting run_pipeline")
    await pipeline.run_pipeline()

if __name__ == "__main__":
    asyncio.run(main())