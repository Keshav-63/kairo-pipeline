# kairo_pipeline/utils/session_tracker.py
import asyncio
from storage.mongodb_handler import MongoDBHandler
from storage.azure_blob import AzureBlobHandler
import logging

logger = logging.getLogger(__name__)

class SessionTracker:
    def __init__(self, mongo_handler: MongoDBHandler, azure_handler: AzureBlobHandler):
        self.mongo = mongo_handler
        self.azure = azure_handler

    async def mark_as_processed(self, session_id, user_id, details):
        await self.mongo.update_session_status(session_id, 'processed', user_id, details)

    async def mark_as_failed(self, session_id, user_id, error):
        await self.mongo.update_session_status(session_id, 'failed', user_id, {'error': str(error)})

    async def get_pending_sessions(self, user_id: str) -> list:
        """
        Compares sessions in Azure with those in MongoDB to find unprocessed sessions.
        """
        try:
            logger.debug(f"Fetching session list for user {user_id} from Azure")
            
            # --- FIX #1: Corrected method name ---
            # Changed from list_unprocessed_sessions to the correct list_user_sessions
            azure_sessions_list = await self.azure.list_user_sessions(user_id)
            azure_sessions = set(azure_sessions_list)
            logger.debug(f"Azure returned {len(azure_sessions)} sessions: {azure_sessions}")

            # --- FIX #2: More robust logic using the dedicated mongo function ---
            processed_sessions = await self.mongo.get_processed_sessions(user_id)
            logger.debug(f"Mongo returned {len(processed_sessions)} processed sessions: {processed_sessions}")
            
            # Determine pending sessions by finding the difference
            pending = list(azure_sessions - processed_sessions)
            logger.info(f"Found {len(pending)} pending sessions for user {user_id}: {pending}")
            
            # Mark newly found sessions as 'pending' in the database
            for session_id in pending:
                await self.mongo.update_session_status(session_id, 'pending', user_id)
            
            return pending
        except Exception as e:
            logger.error(f"Failed to get pending sessions for user {user_id}: {e}", exc_info=True)
            return []