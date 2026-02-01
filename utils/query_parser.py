
import asyncio
from config.logger import logger
from storage.mongodb_handler import AsyncMongoDBHandler
# from kairo_pipeline.utils.time_parser import parse_time_query
import spacy # Assuming spacy is installed and en_core_web_sm is downloaded
import re

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("SpaCy 'en_core_web_sm' model not found. Downloading...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# --- FIX: Define time-related stop words to exclude from keywords ---
TIME_STOP_WORDS = {
    "today", "yesterday", "tomorrow", "morning", "afternoon", "evening", "night",
    "week", "month", "year", "daily", "weekly", "monthly", "annually",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december"
}


def _extract_query_keywords(query_text: str) -> list[str]:
    """
    Extracts relevant keywords, excluding common time-related words.
    """
    doc = nlp(query_text.lower())
    keywords = []
    for token in doc:
        # --- FIX: Check against the time stop words list ---
        if (token.pos_ in ["NOUN", "PROPN", "ADJ"] and 
            not token.is_stop and 
            len(token.lemma_) > 2 and 
            token.lemma_ not in TIME_STOP_WORDS):
            keywords.append(token.lemma_)
            
    keywords.extend([ent.text.lower() for ent in doc.ents 
                     if ent.label_ in ["PERSON", "ORG", "GPE", "NORP", "LOC", "PRODUCT", "EVENT"] 
                     and ent.text.lower() not in TIME_STOP_WORDS])
                     
    return list(set(keywords))


async def parse_query_for_speaker_and_keywords(query_text: str, user_id: str, mongo_handler: AsyncMongoDBHandler):
    """
    Parses a query to find references to enrolled people by relationship or name,
    and also extracts general keywords.

    Returns:
        A tuple containing the cleaned query (without speaker info),
        the identified speaker's name (if any), and a list of extracted keywords.
    """
    cleaned_query = query_text
    speaker_filter = None
    
    try:
        enrollments = await mongo_handler.get_enrollments_for_user(user_id)
        q_lower = query_text.lower()
        
        # Check for both person's name and relationship in the query
        for enrollment in enrollments:
            person_name = (enrollment.get('personName') or '').strip()
            relationship = (enrollment.get('relationship') or '').strip()
            
            if not person_name:
                continue
            
            name_low = person_name.lower()

            # If name appears anywhere, consider it a candidate speaker
            if re.search(rf"\b{re.escape(name_low)}\b", q_lower):
                speaker_filter = person_name
                # Remove common patterns involving the name to get the core query
                # patterns: "what {name} told", "what did {name} say", "{name} said", "what {name} said about X", etc.
                cleaned_query = re.sub(rf"\b(what\s+(did\s+)?{re.escape(name_low)}(\s+(tell|told|say|said))?)\b", "", q_lower, flags=re.I)
                cleaned_query = re.sub(rf"\b{re.escape(name_low)}\b", "", cleaned_query, flags=re.I).strip()
                logger.info(f"Identified speaker '{person_name}' by name in query (flexible match).")
                break
            
            if relationship and re.search(rf"\b{re.escape(relationship.lower())}\b", q_lower):
                speaker_filter = person_name
                cleaned_query = re.sub(rf"\bmy\s+{re.escape(relationship.lower())}\b", "", q_lower, flags=re.I).strip()
                logger.info(f"Identified speaker '{person_name}' by relationship '{relationship}' in query.")
                break

        keywords = _extract_query_keywords(cleaned_query or query_text)
        logger.debug(f"Extracted keywords from query: {keywords}")
        return cleaned_query, speaker_filter, keywords

    except Exception as e:
        logger.error(f"Error parsing query for speaker and keywords: {e}", exc_info=True)
        return query_text, None, [] # Return original query, no speaker, empty keywords on error