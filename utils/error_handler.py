# kairo_pipeline/utils/error_handler.py
import logging
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)

def retry_on_failure(max_retries=3):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)  # Use await for async functions
                except Exception as e:
                    retries += 1
                    logger.error(f"Attempt {retries} failed for {func.__name__}: {str(e)}")
                    if retries == max_retries:
                        logger.error(f"Max retries reached for {func.__name__}")
                        raise
                    await asyncio.sleep(2 ** retries)  # Exponential backoff
        return wrapper
    return decorator