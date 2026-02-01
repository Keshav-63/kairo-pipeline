# kairo_pipeline/main.py
import os
import asyncio
from config.logger import logger
from pipeline import KairoPipeline
from dotenv import load_dotenv

load_dotenv()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)  # Suppress deprecation warnings

async def main():
    logger.info("Initializing KairoPipeline")
    pipeline = KairoPipeline()
    logger.info("KairoPipeline initialized, starting run_pipeline")
    await pipeline.run_pipeline()

if __name__ == "__main__":
    asyncio.run(main())