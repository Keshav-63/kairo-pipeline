# --- START OF FILE main_analytics.py ---
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables (e.g., MongoDB URI, Hugging Face Token)
load_dotenv()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning) # Suppress deprecation warnings

# Import the logger from the kairo_pipeline config
from config.logger import logger

# Import the new analytics pipeline
from analytics_pipeline import AnalyticsPipeline

async def main():
    logger.info("Initializing Kairo Analytics Pipeline")
    pipeline = AnalyticsPipeline()
    logger.info("Kairo Analytics Pipeline initialized, starting run_analytics_pipeline")
    await pipeline.run_analytics_pipeline()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Kairo Analytics Pipeline stopped by user.")
    except Exception as e:
        logger.critical(f"Unhandled critical error in main analytics execution: {e}", exc_info=True)

# --- END OF FILE main_analytics.py ---