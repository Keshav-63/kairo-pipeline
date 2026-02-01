
from datetime import datetime, time
import pytz
from config.logger import logger
from dateparser.search import search_dates

# Set your local timezone (e.g., India Standard Time)
USER_TIMEZONE = pytz.timezone("Asia/Kolkata")

def parse_time_query(query: str):
    """
    Parses a query for time phrases based on the USER_TIMEZONE and returns
    a corresponding UTC timestamp range for the database.
    """
    try:
        now_in_user_tz = datetime.now(USER_TIMEZONE)
        
        # --- FIX: Re-add 'DATE_ORDER': 'MDY' ---
        # This is CRITICAL to fix the "October 7, 2025" bug.
        # It tells the parser to prefer Month-Day-Year format,
        # resolving the ambiguity that caused it to parse "7" as the month.
        # This will NOT break "7 oct 2025" because "oct" is unambiguous.
        settings = {
            'RELATIVE_BASE': now_in_user_tz, 
            'TIMEZONE': USER_TIMEZONE.zone,
            'DATE_ORDER': 'MDY' 
        }
        # --- END FIX ---

        parsed_results = search_dates(query, settings=settings)

        if not parsed_results:
            logger.debug(f"No date/time expressions found in query: '{query}'")
            return None, None

        # --- Sort results to prioritize longest (most specific) match ---
        # "oct 7 2025" (len 11) or "7 oct 2025" (len 10) will be chosen
        # over "to" (len 2) or "on" (len 2).
        sorted_results = sorted(parsed_results, key=lambda x: len(x[0]), reverse=True)
        
        common_ambiguous_words = {"to", "on", "at", "in", "for", "is", "are", "am", "do","time"}

        # Take the longest (best) match
        matched_text, parsed_dt_local = sorted_results[0]

        # If the *best* match is still one of these ambiguous words,
        # it's likely no real date was intended.
        if matched_text.lower() in common_ambiguous_words:
             logger.info(f"Date match was ambiguous ('{matched_text}'). Discarding date filter.")
             return None, None

        logger.info(f"Parsed '{matched_text}' from query, resulting in local datetime: {parsed_dt_local}")

        # Calculate the start and end of that day in the user's local timezone
        start_of_day_local = parsed_dt_local.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day_local = parsed_dt_local.replace(hour=23, minute=59, second=59, microsecond=999999)

        # Convert the local day range to UTC timestamps for the database query
        start_timestamp = int(start_of_day_local.timestamp())
        end_timestamp = int(end_of_day_local.timestamp())

        logger.info(f"Querying for local day: {start_of_day_local.date()}")
        logger.info(f"Generated UTC timestamp range for DB: {start_timestamp} to {end_timestamp}")

        return start_timestamp, end_timestamp

    except Exception as e:
        logger.error(f"Error parsing time from query '{query}': {e}", exc_info=True)
        return None, None