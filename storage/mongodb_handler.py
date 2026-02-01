        
#----------------------------------------------------------------------
import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from datetime import datetime, time, timezone
import pytz
from config.logger import logger
from datetime import datetime, timedelta, time
import numpy as np
from bson import ObjectId
from werkzeug.exceptions import BadRequest
INDIA_TZ = pytz.timezone('Asia/Kolkata')

class AsyncMongoDBHandler:
    def __init__(self, uri, db_name='kairo_db'):
        logger.info(f"Connecting to MongoDB at {uri}")
        self.client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
        try:
            self.client.server_info()
            logger.info("MongoDB async connection successful")
        except Exception as e:
            logger.error(f"MongoDB async connection failed: {str(e)}")
            raise
        self.db = self.client[db_name]
        self.sessions = self.db.sessions
        # Existing transcripts collection (can still be used for raw full transcript if needed)
        self.transcripts = self.db.transcripts 
        # NEW: For storing detailed speaker-diarized segments
        self.transcript_segments = self.db.transcript_segments 
        self.enrollments = self.db.enrollments
        self.logs = self.db.logs
        self.history = self.db.history # For query history
        self.users = self.db.users
        # NEW: Collection for chat sessions
        self.chat_sessions = self.db.chat_sessions
        self.memories_collection = self.db["memories"]
        self.daily_analytics = self.db.daily_analytics
        self.tasks = self.db.tasks
        
        
    # In mongodb_handler.py, add these methods to AsyncMongoDBHandler:
    async def get_chat_history_for_session(self, session_id: str) -> list:
        try:
            cursor = self.history.find({"sessionId": session_id}, {"messages": 1, "_id": 0}).sort("timestamp", 1)
            history = await cursor.to_list(length=100)  # Limit to recent history
            messages = []
            for doc in history:
                messages.extend(doc.get("messages", []))
            logger.debug(f"Retrieved {len(messages)} messages for session {session_id}.")
            return messages
        except Exception as e:
            logger.error(f"Failed to get chat history for {session_id}: {e}", exc_info=True)
            return []

    async def append_to_chat_history(self, session_id: str, user_msg: dict, ai_msg: dict):
        try:
            await self.history.update_one(
                {"sessionId": session_id},
                {"$push": {"messages": {"$each": [user_msg, ai_msg]}}, "$set": {"updatedAt": datetime.utcnow()}},
                upsert=True
            )
            logger.debug(f"Appended messages to history for session {session_id}.")
        except Exception as e:
            logger.error(f"Failed to append to chat history for {session_id}: {e}", exc_info=True)
    
        # --- NEW CHAT HISTORY METHODS ---

    async def create_chat_session(self, user_id: str, title: str = "New Chat"):
        """
        Creates a new chat session for a user.
        """
        session_doc = {
            "user_id": user_id,
            "title": title,
            "created_at": datetime.utcnow(),
            "history": [] # An array of messages {role: 'user'/'assistant', content: '...'}
        }
        result = await self.chat_sessions.insert_one(session_doc)
        logger.info(f"Created new chat session {result.inserted_id} for user {user_id}")
        return str(result.inserted_id)

    async def get_chat_session(self, session_id: str):
        """
        Retrieves a single chat session by its ID.
        """
        return await self.chat_sessions.find_one({"_id": ObjectId(session_id)})

    async def get_user_chat_sessions(self, user_id: str):
        """
        Retrieves all chat sessions for a specific user, sorted by creation date.
        """
        cursor = self.chat_sessions.find(
            {"user_id": user_id,
                # ADDED: This filter only returns chats where the history array is not empty.
            "history": { "$ne": [] }},
            {"history": 0} # Exclude the history for a lighter payload
        ).sort("created_at", -1)
        return await cursor.to_list(length=None)

    async def update_chat_title(self, session_id: str, new_title: str):
        """
        Updates the title of a specific chat session.
        """
        result = await self.chat_sessions.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"title": new_title}}
        )
        if result.modified_count > 0:
            logger.info(f"Updated title for chat session {session_id} to '{new_title}'")
        else:
            logger.warning(f"Could not find chat session {session_id} to update title.")

    async def add_message_to_history(self, session_id: str, role: str, content: str):
        """
        Adds a message to a chat session's history.
        """
        await self.chat_sessions.update_one(
            {"_id": ObjectId(session_id)},
            {"$push": {"history": {"role": role, "content": content, "timestamp": datetime.utcnow()}}}
        )

    # --- METHOD TO GET FULL RAW TRANSCRIPT ---
    async def get_full_raw_transcript(self, session_id: str) -> str:
        """
        Retrieves and concatenates all raw chunk transcripts for a given session.
        """
        try:
            cursor = self.transcripts.find(
                {"sessionId": session_id},
                {"fullText": 1, "_id": 0}
            ).sort("chunkNum", 1)
            
            chunks = await cursor.to_list(length=None)
            full_transcript = " ".join([chunk.get("fullText", "") for chunk in chunks])
            return full_transcript.strip()
        except Exception as e:
            logger.error(f"Failed to retrieve full raw transcript for session {session_id}: {e}")
            return ""
        
        
    async def update_session_status(self, session_id, status, user_id, details=None):
        """
        Updates the status and details of a session without overwriting existing details.
        """
        logger.info(f"Updating session {session_id} for user {user_id} to status: {status}")

        # --- START OF FIX ---
        # Create a dictionary for the fields we want to set.
        update_doc = {
            'status': status,
            'updatedAt': datetime.utcnow()
        }
        
        # If new details are provided, add them to the update document using dot notation.
        # This will merge the new details into the existing 'details' object instead of replacing it.
        if details:
            for key, value in details.items():
                update_doc[f'details.{key}'] = value
        
        # Use the dynamically created update document.
        await self.sessions.update_one(
            {'sessionId': session_id, 'userId': user_id},
            {'$set': update_doc},
            upsert=True
        )
        # --- END OF FIX ---

    async def get_processed_sessions(self, user_id: str) -> set:
        terminal_statuses = ["completed", "archived", "processed"]
        cursor = self.sessions.find(
            {"userId": user_id, "status": {"$in": terminal_statuses}},
            {"sessionId": 1, "_id": 0}
        )
        return {doc["sessionId"] async for doc in cursor}
    
    
    
    # Add these new methods:
    async def save_memory(self, memory_data: dict):
        """Saves a new memory document to the memories collection."""
        try:
            await self.memories_collection.insert_one(memory_data)
            logger.info(f"Successfully saved memory for session {memory_data.get('session_id')}")
        except Exception as e:
            logger.error(f"Failed to save memory for session {memory_data.get('session_id')}: {e}", exc_info=True)

    async def update_memory_with_drive_id(self, session_id: str, drive_id: str):
        """Adds the Google Drive ID to an existing memory document."""
        try:
            await self.memories_collection.update_one(
                {"session_id": session_id},
                {"$set": {"recording_details.drive_id": drive_id}}
            )   
            logger.info(f"Updated memory for session {session_id} with drive_id.")
        except Exception as e:
            logger.error(f"Failed to update memory with drive_id for session {session_id}: {e}", exc_info=True)

    async def get_memories_for_user(self, user_id: str):
        """Retrieves all memories for a given user, sorted by creation date."""
        cursor = self.memories_collection.find({"user_id": user_id}).sort("created_at", -1)
        logger.debug(f"memories of user: {user_id} full memories description endpoint trace catch kar liyaaaa...{cursor}")
        return await cursor.to_list(length=None)

    async def get_memory(self, session_id: str):
        """Retrieves a single memory by session_id."""
        return await self.memories_collection.find_one({"session_id": session_id})

    async def get_transcripts_for_sessions(self, session_ids: list[str]) -> list[str]:
        """
        Retrieves assembled full transcripts for given session IDs from `transcript_segments`
        for LLM context, ensuring chronological order and including speaker labels.
        """
        try:
            if not session_ids:
                return []
            
            logger.debug(f"Querying transcript segments for sessions for LLM context: {session_ids}")
            
            pipeline = [
                {"$match": {"sessionId": {"$in": session_ids}}},
                {"$sort": {"absoluteStartTime": 1}}, # Sort by segment's absolute timestamp
                {"$group": {
                    "_id": "$sessionId",
                    "full_text_segments": {"$push": {"$concat": ["$speaker", ": ", "$text"]}} # Combine speaker and text
                }}
            ]
            
            cursor = self.transcript_segments.aggregate(pipeline)
            
            contexts = []
            async for doc in cursor:
                full_transcript = " ".join(filter(None, doc.get("full_text_segments", [])))
                if full_transcript:
                    contexts.append({
                        "sessionId": doc["_id"],
                        "text": full_transcript.strip()
                    })
            logger.debug(f"Successfully retrieved and assembled full transcripts for {len(contexts)} sessions for LLM context.")
            return contexts
            
        except Exception as e:
            logger.error(f"Failed to retrieve transcripts for LLM context: {e}", exc_info=True)
            return []
        


    # NEW: Method to save the full, raw transcription of a chunk
    async def save_raw_chunk_transcript(self, session_id: str, user_id: str, chunk_num: int,
                                      full_text: str, absolute_chunk_start_time: int, absolute_chunk_end_time: int):
        await self.transcripts.insert_one({
            'sessionId': session_id,
            'userId': user_id,
            'chunkNum': chunk_num,
            'fullText': full_text,
            'absoluteChunkStartTime': absolute_chunk_start_time, # Unix timestamp
            'absoluteChunkEndTime': absolute_chunk_end_time,     # Unix timestamp
            'createdAt': datetime.utcnow()
        })
        logger.debug(f"Saved raw chunk transcript for {session_id}_{chunk_num} to 'transcripts' collection.")

    # NEW: Method to save detailed, speaker-diarized segments
    async def save_transcript_segment(self, session_id: str, user_id: str, chunk_num: int, segment_idx: int,
                                      speaker: str, text: str, start: float, end: float,
                                      keywords: list = None, timestamp: datetime = None):
        if timestamp is None:
            timestamp = datetime.utcnow() # Use the time the segment was recorded if available, otherwise current.
            # Ideally, `timestamp` should come from the session's overall start time + segment offset

        await self.transcript_segments.insert_one({
            'sessionId': session_id,
            'userId': user_id,
            'chunkNum': chunk_num,
            'segmentIdx': segment_idx,
            'speaker': speaker,
            'text': text,
            'start': start, # start time relative to chunk
            'end': end,   # end time relative to chunk
            'absoluteStartTime': timestamp.timestamp() + start, # for Pinecone search
            'absoluteEndTime': timestamp.timestamp() + end,   # for Pinecone search
            'keywords': keywords if keywords is not None else [],
            'createdAt': datetime.utcnow()
        })
        logger.debug(f"Saved segment {session_id}_{chunk_num}_{segment_idx} to transcript_segments.")


    async def save_query_history(self, user_id, query, answer, context_ids):
        await self.history.insert_one({
            'userId': user_id,
            'query': query,
            'answer': answer,
            'contextIds': context_ids,
            'timestamp': datetime.utcnow()
        })

    async def get_distinct_users(self):
        return await self.db.users.distinct('userId')


    async def save_enrollment(self, user_id, person_name, relationship, embedding, sample_url):
        """
        Saves a new voice enrollment to the database.
        """
        await self.enrollments.update_one(
            {'userId': user_id, 'personName': person_name},
            {'$set': {
                'relationship': relationship,
                'embedding': embedding.tolist(),  # Store embedding as a list
                'sampleUrl': sample_url,
                'updatedAt': datetime.utcnow()
            }},
            upsert=True
        )


    async def get_enrollments_for_user(self, user_id: str) -> list:
        enrollments = []
        cursor = self.enrollments.find({'userId': user_id})
        async for doc in cursor:
            if 'embedding' in doc and doc['embedding'] is not None:
                doc['embedding'] = np.array(doc['embedding'])
            enrollments.append(doc)
        return enrollments
    
    
# In storage/mongodb_handler.py, add these methods to AsyncMongoDBHandler

    async def get_self_speaker_label(self, user_id: str) -> str:
        """
        Identifies the 'personName' label for the user themselves based on common relationship/name indicators.
        Prioritizes explicit 'self' labels.
        """
        try:
            # Look for an enrollment with relationship 'self' or 'me'
            self_indicators = ['self', 'i', 'user', 'me']
            enrollments = await self.get_enrollments_for_user(user_id)
            for enrollment in enrollments:
                relationship = enrollment.get('relationship', '').lower()
                if relationship in self_indicators:
                    logger.info(f"Found explicit 'self' label for user {user_id}: {enrollment['personName']}")
                    return enrollment['personName']
            
            # Fallback: check for common indicators in personName or relationship
            indicators = ['my voice', 'me', 'myself']
            for enrollment in enrollments:
                person_name = enrollment.get('personName', '').lower()
                for indicator in indicators:
                    if indicator in person_name:
                        logger.info(f"Found indicator '{indicator}' for user {user_id}'s speaker label: {enrollment['personName']}")
                        return enrollment['personName']

            logger.warning(f"Could not conclusively identify 'self' speaker label for user {user_id}. Returning None.")
            return None
        except Exception as e:
            logger.error(f"Error identifying self speaker label for user {user_id}: {e}", exc_info=True)
            return None

    async def get_unprocessed_sessions_for_analytics(self, user_id: str) -> list:
        """
        Retrieves sessions that are 'completed' by the main pipeline and have not yet
        been successfully processed for analytics.
        """
        try:
            # --- START OF ROBUST FIX ---
            # Find sessions that are 'completed' BUT their 'details.analyticsStatus' is NOT 'complete'.
            # This correctly finds new sessions and sessions that failed in the past, but ignores successful ones.
            cursor = self.sessions.find({
                "userId": user_id,
                "status": "completed",
                "details.analyticsStatus": {"$ne": "complete"} # Use '$ne' (not equal) for robustness
            })
            # --- END OF ROBUST FIX ---
            return await cursor.to_list(length=None)
        except Exception as e:
            logger.error(f"Failed to get unprocessed sessions for analytics for user {user_id}: {e}", exc_info=True)
            return []


    async def update_session_analytics_status(self, session_id: str, user_id: str, status: str = 'complete', details: dict = None):
        """
        Marks a session with a specific analytics status ('complete' or 'failed').
        """
        # --- START OF ROBUST FIX ---
        # Use a dedicated status field as requested.
        update_doc = {
            'details.analyticsStatus': status,
            'details.analyticsProcessedAt': datetime.utcnow() # Keep timestamp for logging/debugging
        }
        if status == 'failed' and details:
            update_doc['details.analyticsError'] = details.get('error')
        # --- END OF ROBUST FIX ---

        result = await self.sessions.update_one(
            {"sessionId": session_id, "userId": user_id},
            {"$set": update_doc}
        )
        if result.modified_count > 0:
            logger.info(f"Session {session_id} marked as analytics '{status}' for user {user_id}")


    # async def upsert_daily_analytics(self, user_id: str, analytics_date: datetime.date, metrics: dict):
    #     """
    #     Upserts daily behavioral analytics data for a user.
    #     `analytics_date` should be a date object (e.g., datetime.date(2023, 10, 26)).
    #     """
    #     try:
    #         day_start_utc = datetime.combine(analytics_date, time.min, tzinfo=pytz.utc)

    #         update_query = {"$inc": {}}
    #         for metric, value in metrics.items():
    #             if isinstance(value, (int, float)):
    #                 update_query["$inc"][metric] = value

    #         if not update_query["$inc"]:
    #             logger.warning(f"No metrics to upsert for user {user_id} on {analytics_date}")
    #             return

    #         update_query["$set"] = {"updatedAt": datetime.utcnow()}

    #         await self.daily_analytics.update_one(
    #             {"user_id": user_id, "date": day_start_utc},
    #             update_query,
    #             upsert=True
    #         )
    #         logger.info(f"Upserted daily analytics for user {user_id} on {analytics_date}")
    #     except Exception as e:
    #         logger.error(f"Failed to upsert daily analytics for user {user_id}: {e}", exc_info=True)
    
    async def upsert_daily_analytics(self, user_id: str, session_date, metrics: dict):
        try:
        # Create datetime at start of day in Indian timezone
            india_date = datetime.combine(session_date, time.min)
            india_datetime = INDIA_TZ.localize(india_date)
        
        # Log the timezone conversions for debugging
            logger.debug(f"Original date: {session_date}")
            logger.debug(f"India datetime: {india_datetime}")
            logger.debug(f"UTC datetime: {india_datetime.astimezone(pytz.UTC)}")

            update_data = {
                "$set": {
                    "user_id": user_id,
                    "date": india_datetime,  # Store the Indian timezone datetime
                    "updatedAt": datetime.now(INDIA_TZ)
                },
                "$inc": metrics
            }
        
            result = await self.db.daily_analytics.update_one(
                {
                    "user_id": user_id,
                    "date": india_datetime
                },
                update_data,
                upsert=True
            )
            return result
        except Exception as e:
            logger.error(f"Error upserting daily analytics: {e}")
            raise

    async def get_daily_analytics_summary(self, user_id: str) -> list:
        """
        Retrieves a summary of daily analytics for a user, sorted by date,
        and calculates the ratio dynamically.
        """
        cursor = self.daily_analytics.find(
            {"user_id": user_id},
            # --- FIX: Fetch the raw totals instead of the stored ratio ---
            {"_id": 0, "date": 1, "socialTime": 1, "totalSpeakingTime": 1, "totalListeningTime": 1}
        ).sort("date", -1)
        
        summary_list = await cursor.to_list(length=30)

        # --- FIX: Calculate the ratio in the backend before sending ---
        for entry in summary_list:
            speak_time = entry.get("totalSpeakingTime", 0)
            listen_time = entry.get("totalListeningTime", 0)
            if listen_time > 0:
                entry["speakingToListeningRatio"] = speak_time / listen_time
            else:
                entry["speakingToListeningRatio"] = speak_time
        
        return summary_list

    async def get_daily_analytics_by_date(self, user_id: str, date_str: str) -> dict:
        """
        Retrieves detailed daily analytics and calculates the ratio dynamically.
        """
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            day_start_utc = datetime.combine(target_date, time.min, tzinfo=pytz.utc)
            
            report = await self.daily_analytics.find_one(
                {"user_id": user_id, "date": day_start_utc},
                {"_id": 0}
            )
            
            # --- FIX: Calculate and add the ratio to the final report ---
            if report:
                speak_time = report.get("totalSpeakingTime", 0)
                listen_time = report.get("totalListeningTime", 0)
                if listen_time > 0:
                    report["speakingToListeningRatio"] = speak_time / listen_time
                else:
                    report["speakingToListeningRatio"] = speak_time

            return report
        except ValueError:
            logger.error(f"Invalid date string format: {date_str}. Expected YYYY-MM-DD.")
            raise BadRequest("Invalid date format. Please use YYYY-MM-DD.")
        except Exception as e:
            logger.error(f"Failed to retrieve daily analytics for {user_id} on {date_str}: {e}", exc_info=True)
            return None
        
        
# In storage/mongodb_handler.py, inside the AsyncMongoDBHandler class

    async def get_segments_for_session(self, session_id: str) -> list:
        """
        Retrieves all transcript segments for a given session, sorted chronologically.
        """
        try:
            cursor = self.transcript_segments.find(
                {"sessionId": session_id}
            ).sort("start", 1) # Sort by the segment start time
            return await cursor.to_list(length=None)
        except Exception as e:
            logger.error(f"Failed to get segments for session {session_id}: {e}", exc_info=True)
            return []



# Remainder & task
    async def save_task(self, task_data: dict):
        """Saves a new task document to the tasks collection."""
        try:
            result = await self.tasks.insert_one(task_data)
            logger.info(f"Saved task {result.inserted_id} for user {task_data.get('user_id')}")
            # --- MODIFICATION: Return the inserted ID ---
            return result.inserted_id
        except Exception as e:
            logger.error(f"Failed to save task '{task_data.get('task_description')}': {e}", exc_info=True)


    # async def get_tasks_by_status(self, user_id: str, status: str):
    #     """Retrieves tasks for a user, filtered by status."""
    #     cursor = self.tasks.find({"user_id": user_id, "status": status}).sort("due_date_time", 1)
    #     return await cursor.to_list(length=None)
    async def get_tasks_by_status(self, user_id: str, status: str = 'pending'):
        """Get tasks for a user filtered by status."""
        try:
            query = {
                "user_id": user_id,
                "is_task": True
            }
        
        # Handle different status filters
            if status == 'pending':
            # Include both pending and scheduled tasks without completion
                query["status"] = {"$in": ["pending", "scheduled"]}
            elif status == 'complete':
                query["status"] = "complete"
            
            cursor = self.db.tasks.find(query).sort("created_at", -1)
            tasks = await cursor.to_list(length=None)
        
            logger.debug(f"Found {len(tasks)} {status} tasks for user {user_id}")
            return tasks
        
        except Exception as e:
            logger.error(f"Error getting tasks for user {user_id}: {e}")
            return []

    async def update_task_status(self, task_id: str, status: str):
        """Updates the status of a specific task by its ObjectId."""
        return await self.tasks.update_one(
            {"_id": ObjectId(task_id)},
            {"$set": {"status": status, "completed_at": datetime.utcnow()}}
        )


    async def update_user_tokens(self, user_id: str, access_token: str, refresh_token: str = None):
        """
        NEW: Updates user's OAuth tokens after refresh.
        """
        try:
            update_doc = {
                "accessToken": access_token,
                "updatedAt": datetime.utcnow()
            }
        
            # Only update refresh_token if provided (it's not always returned)
            if refresh_token:
                update_doc["refreshToken"] = refresh_token
        
            await self.users.update_one(
                {"userId": user_id},
                {"$set": update_doc}
            )
        
            logger.info(f"Updated OAuth tokens for user {user_id}")
        
        except Exception as e:
            logger.error(f"Failed to update tokens for user {user_id}: {e}", exc_info=True)
            
            
    async def get_segments_for_chunks(self, chunk_identifiers: list) -> list:
        """
        Fetches all transcript segments for a given list of (sessionId, chunkNum) tuples.
        
        Args:
            chunk_identifiers: A list of tuples, e.g., [('session_1', 0), ('session_1', 1)]
        
        Returns:
            A list of all segment documents from those specific chunks.
        """
        if not chunk_identifiers:
            return []
        logger.debug(f"get_segments_for_chunks : {chunk_identifiers}")
        
        # Create a list of $or conditions
        or_conditions = [
            {"sessionId": str(sid), "chunkNum": int(chunk_num)} 
            for sid, chunk_num in chunk_identifiers
        ]
        
        query = {"$or": or_conditions}
        logger.debug(f"get_segments_for_chunks query.......... {query}")
        try:
            segments_cursor = self.db.transcript_segments.find(query)
            segments = await segments_cursor.to_list(length=None) # Get all matching
            logger.info(f"Fetched {len(segments)} segments from {len(chunk_identifiers)} chunks.")
            logger.debug(f"Fetched {segments} segments from {len(chunk_identifiers)} chunks.")
            
            return segments
        except Exception as e:
            logger.error(f"Failed to fetch segments for chunks: {e}", exc_info=True)
            return []
        
        
    async def _extract_session_ids(self, results: list) -> list[str]:
        """Extracts unique session IDs from Pinecone search results."""
        session_ids = set()
        for hit in results:
            session_id = hit.get('sessionId')
            if session_id:
                session_ids.add(session_id)
        return list(session_ids)
        
# --- FIX: 2 (Add the new method) ---
    async def update_task_with_calendar_id(self, task_mongo_id, calendar_event_id):
        """
        Updates a task document with the Google Calendar event ID and sets
        status to 'scheduled'.
        """
        try:
            if not isinstance(task_mongo_id, ObjectId):
                task_mongo_id = ObjectId(task_mongo_id)
            
            await self.tasks.update_one(
                {'_id': task_mongo_id},
                {'$set': {
                    'googleCalendarEventId': calendar_event_id, 
                    'status': 'scheduled'
                    }}
            )
            logger.info(f"Updated task {task_mongo_id} with calendar event ID: {calendar_event_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update task {task_mongo_id} with calendar ID: {e}", exc_info=True)
            return False
        
# --- FIX: 3 (Replace `update_user_tokens` with `update_user_credentials`) ---
    async def update_user_credentials(self, user_id, new_credentials_dict):
        """
        Updates the user's credentials after a token refresh.
        This implementation updates the root fields to match your
        SyncHandler's `get_user_credentials` method.
        """
        try:
            update_doc = {
                "accessToken": new_credentials_dict['token'],
                "updatedAt": datetime.utcnow()
            }
        
            # Add expiry if it exists in the new credentials
            if new_credentials_dict.get('expiry'):
                 # Convert to datetime object if it's an ISO string
                if isinstance(new_credentials_dict['expiry'], str):
                    update_doc["expiry"] = datetime.fromisoformat(new_credentials_dict['expiry'])
                else:
                    update_doc["expiry"] = new_credentials_dict['expiry']

            # Only update refresh_token if it was returned
            if new_credentials_dict.get('refresh_token'):
                update_doc["refreshToken"] = new_credentials_dict['refresh_token']
        
            # Use 'userId' as the key, matching your SyncHandler
            result = await self.users.update_one(
                {"userId": user_id},
                {"$set": update_doc}
            )
            
            if result.matched_count == 0:
                 logger.error(f"Failed to update credentials: User {user_id} not found.")
                 return False
        
            logger.info(f"Successfully updated OAuth credentials for user {user_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update credentials for user {user_id}: {e}", exc_info=True)
            return False

# Set MongoDBHandler alias after defining AsyncMongoDBHandler
MongoDBHandler = AsyncMongoDBHandler


class SyncMongoDBHandler:
    def __init__(self, uri, db_name='kairo_db'):
        logger.info(f"Connecting to MongoDB at {uri}")
        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        try:
            self.client.server_info()
            logger.info("MongoDB sync connection successful")
        except Exception as e:
            logger.error(f"MongoDB sync connection failed: {str(e)}")
            raise
        self.db = self.client[db_name]
        self.sessions = self.db.sessions
        self.transcripts = self.db.transcripts
        self.transcript_segments = self.db.transcript_segments # NEW
        self.enrollments = self.db.enrollments
        self.logs = self.db.logs
        self.history = self.db.history
        self.users = self.db.users
        self.memories_collection = self.db["memories"]
        self.daily_analytics = self.db.daily_analytics # NEW
        
    def get_user_credentials(self, user_id: str) -> dict:
        """
        Retrieves a user's Google OAuth credentials from the database.
        """
        try:
            user = self.users.find_one({"userId": user_id})
        
            if not user:
                logger.error(f"No user document found for userId: {user_id}")
                return None
        
            # CRITICAL: Validate all required OAuth2 fields exist
            if not user.get("accessToken"):
                logger.error(f"User {user_id} has no access token")
                return None
            
            if not user.get("refreshToken"):
                logger.error(
                    f"User {user_id} has no refresh token. "
                    f"User must re-authenticate with prompt=consent."
                )
                return None
        
            # IMPROVED: Return complete OAuth2 credentials
            credentials = {
                "token": user["accessToken"],
                "refresh_token": user["refreshToken"],
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_id": os.environ.get("GOOGLE_CLIENT_ID"),
                "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET"),
                "scopes": user.get("scopes", ["https://www.googleapis.com/auth/drive"])
            }
        
            # Validate environment variables
            if not credentials["client_id"] or not credentials["client_secret"]:
                logger.error(
                    "GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET environment variables not set."
                )
                return None
        
            logger.debug(f"Successfully retrieved complete credentials for user {user_id}")
            return credentials
        
        except Exception as e:
            logger.error(f"Failed to retrieve user credentials: {e}", exc_info=True)
            return None
        
    # --- ADD THIS NEW METHOD ---
    # def get_all_user_sessions(self, user_id: str) -> list:
    #     """
    #     Retrieves all session documents for a given user, sorted by creation date.
    #     """
    #     try:
    #         # The 'sessions' collection contains the metadata for each recording
    #         cursor = self.sessions.find({"userId": user_id}).sort("createdAt", -1)
    #         logger.info(f"Found sessions for user {user_id} in MongoDB.")
    #         return list(cursor)
    #     except Exception as e:
    #         logger.error(f"Failed to retrieve all sessions for user {user_id}: {e}", exc_info=True)
    #         return []
    def get_all_user_sessions(self, user_id: str, sort_order: int = -1) -> list:
        """
        Gets all sessions for a user, sorted by sessionId.
    
        Args:
            user_id: The user's ID
            sort_order: 1 for ascending, -1 for descending (default: -1 for newest first)
        """
        try:
        # Fix: Change user_id to userId to match your MongoDB schema
            cursor = self.db.sessions.find(
                {"userId": user_id},  # Changed from user_id to userId
                {"_id": 1, "sessionId": 1, "details": 1}
            ).sort("sessionId", sort_order)
        
            sessions = list(cursor)
            logger.info(f"Found {len(sessions)} sessions for user {user_id}")
            return sessions
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []

    def update_session_status(self, session_id, status, user_id, details=None):
        self.sessions.update_one(
            {'sessionId': session_id, 'userId': user_id},
            {'$set': {'status': status, 'updatedAt': datetime.utcnow(), 'details': details or {}}},
            upsert=True
        )

    def get_processed_sessions(self, user_id: str) -> set:
        terminal_statuses = ["completed", "archived", "processed"]
        cursor = self.sessions.find(
            {"userId": user_id, "status": {"$in": terminal_statuses}},
            {"sessionId": 1, "_id": 0}
        )
        return {doc["sessionId"] for doc in cursor}

    def get_transcripts_for_sessions(self, session_ids: list[str]) -> list[str]:
        """
        Retrieves assembled full transcripts for given session IDs from `transcript_segments`.
        This is for LLM context, so it should be chronological and complete.
        """
        try:
            if not session_ids:
                return []
            
            logger.debug(f"Querying transcript segments for sessions for LLM context (Sync): {session_ids}")
            
            pipeline = [
                {"$match": {"sessionId": {"$in": session_ids}}},
                {"$sort": {"absoluteStartTime": 1}}, # Sort by absolute start time
                {"$group": {
                    "_id": "$sessionId",
                    "full_text_segments": {"$push": {"$concat": ["$speaker", ": ", "$text"]}} # Combine speaker and text
                }}
            ]
            
            cursor = self.transcript_segments.aggregate(pipeline)
            
            contexts = []
            for doc in cursor:
                full_transcript = " ".join(filter(None, doc.get("full_text_segments", [])))
                if full_transcript:
                    contexts.append(full_transcript.strip())
            
            logger.debug(f"Successfully retrieved and assembled full transcripts for {len(contexts)} sessions for LLM context (Sync).")
            return contexts
            
        except Exception as e:
            logger.error(f"Failed to retrieve transcripts for LLM context (Sync): {e}", exc_info=True)
            return []

    def update_chat_title_sync(self, session_id: str, new_title: str):
        """
        Updates the title of a specific chat session synchronously.
        """
        self.chat_sessions.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"title": new_title}}
        )
        
    # NEW: Method to save the full, raw transcription of a chunk (Sync version)
    def save_raw_chunk_transcript(self, session_id: str, user_id: str, chunk_num: int,
                                      full_text: str, absolute_chunk_start_time: int, absolute_chunk_end_time: int):
        self.transcripts.insert_one({
            'sessionId': session_id,
            'userId': user_id,
            'chunkNum': chunk_num,
            'fullText': full_text,
            'absoluteChunkStartTime': absolute_chunk_start_time,
            'absoluteChunkEndTime': absolute_chunk_end_time,
            'createdAt': datetime.utcnow()
        })
        logger.debug(f"Saved raw chunk transcript for {session_id}_{chunk_num} to 'transcripts' collection (Sync).")
        
    def save_transcript_segment(self, session_id: str, user_id: str, chunk_num: int, segment_idx: int,
                                  speaker: str, text: str, start: float, end: float,
                                  keywords: list = None, timestamp: datetime = None):
        if timestamp is None:
            timestamp = datetime.utcnow() # Use the time the segment was recorded if available, otherwise current.

        self.transcript_segments.insert_one({
            'sessionId': session_id,
            'userId': user_id,
            'chunkNum': chunk_num,
            'segmentIdx': segment_idx,
            'speaker': speaker,
            'text': text,
            'start': start,
            'end': end,
            'absoluteStartTime': timestamp.timestamp() + start,
            'absoluteEndTime': timestamp.timestamp() + end,
            'keywords': keywords if keywords is not None else [],
            'createdAt': datetime.utcnow()
        })
        logger.debug(f"Saved segment {session_id}_{chunk_num}_{segment_idx} to transcript_segments (Sync).")

    def save_query_history(self, user_id, query, answer, context_ids):
        self.history.insert_one({
            'userId': user_id,
            'query': query,
            'answer': answer,
            'contextIds': context_ids,
            'timestamp': datetime.utcnow()
        })

    def get_distinct_users(self):
        return list(self.db.users.distinct('userId'))
        
    def get_session_details(self, session_id: str, user_id: str) -> dict:
        """
        Retrieves the Google Drive file ID and encryption key for a given session.
        """
        try:
            logger.debug(f"Retrieving session details for session: {session_id}")
            session_details = self.sessions.find_one(
                {"sessionId": session_id, "userId": user_id},
                {"details.driveId": 1, "details.encryptionKey": 1, "_id": 0}
            )
            return session_details['details'] if session_details else None
        except Exception as e:
            logger.error(f"Failed to retrieve session details: {e}", exc_info=True)
            return None
    
    def save_enrollment(self, user_id, person_name, relationship, embedding, sample_url):
        """
        Saves a new voice enrollment to the database synchronously.
        """
        self.enrollments.update_one(
            {'userId': user_id, 'personName': person_name},
            {'$set': {
                'relationship': relationship,
                'embedding': embedding.tolist(),
                'sampleUrl': sample_url,
                'updatedAt': datetime.utcnow()
            }},
            upsert=True
        )

# Add these new methods:
    def save_memory(self, memory_data: dict):
        """Saves a new memory document to the memories collection."""
        try:
            self.memories_collection.insert_one(memory_data)
            logger.info(f"Successfully saved memory for session {memory_data.get('session_id')}")
        except Exception as e:
            logger.error(f"Failed to save memory for session {memory_data.get('session_id')}: {e}", exc_info=True)

    def update_memory_with_drive_id(self, session_id: str, drive_id: str):
        """Adds the Google Drive ID to an existing memory document."""
        try:
            self.memories_collection.update_one(
                {"session_id": session_id},
                {"$set": {"recording_details.drive_id": drive_id}}
            )
            logger.info(f"Updated memory for session {session_id} with drive_id.")
        except Exception as e:
            logger.error(f"Failed to update memory with drive_id for session {session_id}: {e}", exc_info=True)

    def get_memories_for_user(self, user_id: str):
        """Retrieves all memories for a given user, sorted by creation date."""
        cursor = self.memories_collection.find({"user_id": user_id}).sort("created_at", -1)
        return cursor.to_list(length=None)

    def get_memory(self, session_id: str):
        """Retrieves a single memory by session_id."""
        return self.memories_collection.find_one({"session_id": session_id})
    

    # Sync version:

    def get_enrollments_for_user(self, user_id: str) -> list:
        """
        Retrieves all enrollments for a given user.
        FIXED: Handles multiple embeddings per enrollment (new format) and backward compatible with old format.
        """
        enrollments = []
        cursor = self.enrollments.find({'userId': user_id})
    
        for doc in cursor:
            # FIXED: Handle new 'embeddings' field (plural - list of embeddings)
            if 'embeddings' in doc and doc['embeddings']:
                # New format: multiple embeddings stored as list of lists
                doc['embeddings'] = [np.array(emb) for emb in doc['embeddings']]
                logger.debug(
                    f"Loaded {len(doc['embeddings'])} embeddings for "
                    f"{doc.get('personName')} (new format)"
                )
                enrollments.append(doc)
            
            # Backward compatibility: Handle old 'embedding' field (singular)
            elif 'embedding' in doc and doc['embedding']:
                # Old format: single embedding - convert to list for consistency
                doc['embeddings'] = [np.array(doc['embedding'])]
                logger.debug(
                    f"Loaded 1 embedding for {doc.get('personName')} (old format, converted)"
                )
                # Remove old field to avoid confusion
                doc.pop('embedding', None)
                enrollments.append(doc)
            else:
                logger.warning(
                    f"Enrollment for {doc.get('personName')} has no embeddings. Skipping."
                )
    
        logger.info(f"Retrieved {len(enrollments)} enrollments for user {user_id}")
        return enrollments