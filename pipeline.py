import asyncio
import os
import shutil
import sys
import whisper
import yaml
from config.logger import logger
from core.diarization import DiarizationPipeline
from core.preprocess import preprocess_audio
from core.voice_recognition import VoiceRecognizer
from storage.azure_blob import AzureBlobHandler
from storage.google_drive import GoogleDriveHandler # Updated
from storage.mongodb_handler import MongoDBHandler, SyncMongoDBHandler
from storage.pinecone_handler import PineconeHandler
from utils.error_handler import retry_on_failure
from utils.session_tracker import SessionTracker
from datetime import datetime, timedelta 
import soundfile as sf
import pytz # --- NEW: IMPORT FOR TIMEZONE-AWARE 'NOW' ---

# Import for VAD and Transcription
from core.vad_transcribe import process_and_transcribe_vad_whisper 
import torch

# For Keyword Extraction
import spacy
import subprocess 

# --- Imports for New Features ---
from core.memories_generator import MemoriesGenerator
from utils.langchain_llm_handler import LangchainLLMHandler
from core.task_detector import TaskDetector
from storage.google_calendar import GoogleCalendarHandler # NEW

# Download 'en_core_web_sm' model if not already present
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("SpaCy 'en_core_web_sm' model not found. Downloading...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


with open('config/settings.yaml', 'r') as f:
    config = yaml.safe_load(f)

class KairoPipeline:
    def __init__(self):
        self.azure = AzureBlobHandler(config['azure']['account_name'], config['azure']['account_key'], config['azure']['container'])
        self.mongo = MongoDBHandler(config['mongodb']['uri'])
        self.mongo_sync = SyncMongoDBHandler(config['mongodb']['uri'])
        self.pinecone = PineconeHandler(config['pinecone']['api_key'], config['pinecone']['index_name'])
        self.diarization_pipeline = DiarizationPipeline(config['hf_token'])
        self.voice_recognizer = VoiceRecognizer()
        self.tracker = SessionTracker(self.mongo, self.azure)

        config_path = os.path.join(os.path.dirname(__file__), 'config', 'settings.yaml')
        self.llm_handler = LangchainLLMHandler(config_path)
        self.memories_generator = MemoriesGenerator(self.llm_handler)
        self.task_detector = TaskDetector(self.llm_handler) 

        logger.info("Initializing Silero VAD model...")
        self.vad_model, self.vad_utils = self._init_silero_vad()
        logger.info("Silero VAD model initialized.")
        
        logger.info("Loading 'medium' Whisper model for transcription...")
        self.whisper_model = whisper.load_model("medium", device=self.diarization_pipeline.device)
        logger.info("Whisper model loaded successfully.")

    def _init_silero_vad(self):
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False, 
                onnx=False
            )
            return model, utils
        except Exception as e:
            logger.critical(f"Failed to initialize Silero VAD model: {e}", exc_info=True)
            raise

    def _extract_keywords(self, text: str) -> list[str]:
        if not text:
            return []
        try:
            doc = nlp(text)
            keywords = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "NORP", "LOC", "PRODUCT", "EVENT"]]
            keywords.extend([token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop and len(token.lemma_) > 2])
            return list(set([k.lower() for k in keywords]))
        except Exception as e:
            logger.error(f"Keyword extraction failed for text: {text[:50]}... Error: {e}", exc_info=True)
            return []

    # @retry_on_failure(max_retries=3)
    async def process_session(self, session_id, user_id):
        logger.info(f"--- Starting processing for session: {session_id} ---")
        temp_dir = f"temp_{session_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        drive_handler = None
        calendar_handler = None
        
        try:
            # --- Fetch user credentials for Google APIs ---
            user_credentials = self.mongo_sync.get_user_credentials(user_id)
            if not user_credentials:
                logger.error(f"Could not find credentials for user {user_id}. Cannot use Google APIs. Skipping session.")
                await self.mongo.update_session_status(session_id, 'failed', user_id, {'error': 'User credentials not found for Google APIs.'})
                return
            
            # --- Initialize Google Drive Handler ---
            try:
                drive_handler = GoogleDriveHandler(
                    user_credentials,
                    mongo_handler=self.mongo, # Pass async handler
                    user_id=user_id
                )
                await drive_handler.initialize_service()
                
            except (ValueError, IOError) as e:
                logger.error(
                    f"Failed to initialize Google Drive handler for user {user_id}: {e}. "
                    f"User may need to re-authenticate."
                )
                await self.mongo.update_session_status(
                    session_id,
                    "failed",
                    user_id,
                    {"error": f"OAuth2 error (Drive): {str(e)}. Re-authentication required."}
                )
                return

            # --- Initialize Google Calendar Handler ---
            try:
                calendar_handler = GoogleCalendarHandler(
                    user_credentials,
                    mongo_handler=self.mongo, # Pass async handler
                    user_id=user_id
                )
                await calendar_handler.initialize_service()
                logger.info(f"Google Calendar handler initialized for user {user_id}")
                
            except (ValueError, IOError) as e:
                logger.warning(
                    f"Failed to initialize Google Calendar handler for user {user_id}: {e}. "
                    f"This is often a MISSING SCOPE. Calendar features will be disabled."
                )
                calendar_handler = None

            # --- End of Handler Initialization ---

            await self.mongo.update_session_status(session_id, 'processing', user_id)
            
            enrollments = await self.mongo.get_enrollments_for_user(user_id)
            
            self_speaker_label = await self.mongo.get_self_speaker_label(user_id)
            if not self_speaker_label:
                logger.warning(f"Could not identify 'self' speaker label for user {user_id}. Task detection will be skipped for this session.")
            
            chunks, metadata = await self.azure.download_session(session_id)
            if not chunks or not metadata:
                raise ValueError("Session download failed or returned no data.")
            
            processed_audio_paths = []
            all_segments_for_pinecone = []
            full_session_transcript_parts = [] 


            session_start_iso = metadata.get('startTime')
            if session_start_iso:
                try:
                    session_start_time_dt = datetime.fromisoformat(session_start_iso.replace('Z', '+00:00'))
                except ValueError: 
                    logger.warning(f"Could not parse session startTime '{session_start_iso}'. Using current UTC time.")
                    session_start_time_dt = datetime.utcnow().replace(tzinfo=pytz.utc)
            else:
                session_start_time_dt = datetime.utcnow().replace(tzinfo=pytz.utc)
            
            session_start_unix_timestamp = session_start_time_dt.timestamp()

            current_session_offset = 0 
            for i, chunk_data in enumerate(chunks):
                logger.info(f"--- Processing Chunk {i + 1} of {len(chunks)} ---")
                original_path = os.path.join(temp_dir, f"original_chunk_{i}.wav")
                preprocessed_path = os.path.join(temp_dir, f"preprocessed_chunk_{i}.wav")
                
                with open(original_path, 'wb') as f:
                    f.write(chunk_data)

                if not preprocess_audio(original_path, preprocessed_path): 
                    logger.error(f"Preprocessing failed for chunk {i}, skipping.")
                    continue
                
                vad_transcription_result = process_and_transcribe_vad_whisper(
                    preprocessed_path, self.vad_model, self.vad_utils, self.whisper_model, 
                    padding_ms=config.get('vad_padding_ms', 400) 
                )
                
                if not vad_transcription_result or not vad_transcription_result.get("text"):
                    logger.info(f"VAD & Transcription for chunk {i} produced no processable speech. Skipping.")
                    continue

                # --- FIX: Apply Kairo replacement IMMEDIATELY ---
                raw_full_chunk_text = vad_transcription_result.get("text", "").strip()
                if raw_full_chunk_text:
                    full_chunk_text = raw_full_chunk_text.replace("Cairo", "Kairo").replace("cairo", "Kairo").replace("CAIRO", "Kairo")
                    # Also update the result object itself if it's used directly later (like in diarization)
                    if vad_transcription_result.get("text"):
                         vad_transcription_result["text"] = full_chunk_text
                    logger.debug(f"Applied Kairo correction. Original: '{raw_full_chunk_text[:50]}...' Corrected: '{full_chunk_text[:50]}...'")
                else:
                    full_chunk_text = "" # Ensure it's an empty string if no text
                # --- END FIX ---
                
                if not full_chunk_text:
                    logger.warning(f"Full text for chunk {i} is empty after VAD & Transcription. Skipping save to 'transcripts'.")
                    continue
                
                try:
                    chunk_audio_info = sf.info(preprocessed_path)
                    chunk_duration = chunk_audio_info.duration
                except Exception as e:
                    logger.warning(f"Could not get precise duration for chunk {i} from {preprocessed_path}. Estimating 10 seconds. Error: {e}")
                    chunk_duration = 10 

                absolute_chunk_start_time = int(session_start_unix_timestamp + current_session_offset)
                absolute_chunk_end_time = int(session_start_unix_timestamp + current_session_offset + chunk_duration)
                await self.mongo.save_raw_chunk_transcript(
                    session_id=session_id,
                    user_id=user_id,
                    chunk_num=i,
                    full_text=full_chunk_text,
                    absolute_chunk_start_time=absolute_chunk_start_time,
                    absolute_chunk_end_time=absolute_chunk_end_time
                )

                diarized_segments = self.diarization_pipeline.process(
                    preprocessed_path, vad_transcription_result, enrollments 
                )
                
                if not diarized_segments:
                    logger.info(f"Diarization for chunk {i} produced no processable speech segments. Skipping.")
                    continue

                for seg_idx, segment in enumerate(diarized_segments):
                    text = segment.get('text', '').strip()
                    if text:
                        keywords = self._extract_keywords(text)
                        
                        absolute_start_time = session_start_unix_timestamp + segment['start']+ current_session_offset
                        absolute_end_time = session_start_unix_timestamp + segment['end']+ current_session_offset

                        await self.mongo.save_transcript_segment(
                            session_id=session_id,
                            user_id=user_id,
                            chunk_num=i,
                            segment_idx=seg_idx,
                            speaker=segment['speaker'],
                            text=text,
                            start=segment['start'],
                            end=segment['end'], 
                            keywords=keywords,
                            timestamp=datetime.fromtimestamp(absolute_start_time, tz=pytz.utc)
                        )
                        
                        pinecone_record = {
                            "_id": f"{session_id}_{i}_{seg_idx}",
                            "text": text,
                            "sessionId": session_id,
                            "userId": user_id,
                            "chunkNum": i,
                            "speaker": segment['speaker'],
                            "start": float(segment['start']),
                            "end": float(segment['end']),
                            "absoluteStartTime": int(absolute_start_time), 
                            "absoluteEndTime": int(absolute_end_time),     
                            "createdAt": int(datetime.utcnow().timestamp()),
                            "keywords": keywords,
                        }
                        all_segments_for_pinecone.append(pinecone_record)
                        
# --- START OF TASK DETECTION & CALENDAR LOGIC ---
                        if segment['speaker'] == self_speaker_label:
                            detected_task = await self.task_detector.detect_task(user_id, text)
                            if detected_task:
                                detected_task['user_id'] = user_id
                                detected_task['session_id'] = session_id
                                detected_task['source_segment_id'] = f"{session_id}_{i}_{seg_idx}"
                                if 'status' not in detected_task: detected_task['status'] = 'pending'
                                if 'created_at' not in detected_task: detected_task['created_at'] = datetime.utcnow()

                                inserted_id = await self.mongo.save_task(detected_task)

                                if inserted_id and calendar_handler:
                                    task_desc = detected_task.get('task_description')
                                    task_due_str = detected_task.get('due_date_time') # Use the key from DB

                                    if task_desc and task_due_str:
                                        try:
                                            logger.info(f"Task '{task_desc}' has due date string '{task_due_str}'. Attempting to parse and create calendar event.")

                                            # --- FIX: Ensure start_dt is timezone-aware ---
                                            start_dt = None # Initialize
                                            try:
                                                # Attempt parsing, handles both aware and naive strings
                                                start_dt = datetime.fromisoformat(task_due_str)
                                            except ValueError as ve:
                                                # Log the parsing error specifically
                                                logger.error(f"Task '{task_desc}' due date string '{task_due_str}' could not be parsed into a valid ISO datetime: {ve}. Skipping calendar.")
                                                continue # Skip calendar creation for this task if parsing fails

                                            # *** AFTER parsing, check if it's naive ***
                                            if start_dt.tzinfo is None:
                                                 logger.warning(f"Parsed due date '{task_due_str}' as naive datetime. Assuming UTC based on LLM prompt instructions.")
                                                 # Make it timezone-aware by assuming UTC
                                                 start_dt = start_dt.replace(tzinfo=pytz.utc)
                                            # --- END FIX ---

                                            end_dt = start_dt + timedelta(minutes=30)
                                            tz_str = str(start_dt.tzinfo) # Should now always have tzinfo

                                            # --- DYNAMIC REMINDER LOGIC ---
                                            now_utc = datetime.now(pytz.utc) # This is timezone-aware

                                            # Now subtraction should work as both are timezone-aware
                                            delta = start_dt - now_utc
                                            minutes_before_now = int(delta.total_seconds() / 60)

                                            overrides_list = [
                                                {'method': 'popup', 'minutes': 0},
                                                {'method': 'popup', 'minutes': 30},
                                                {'method': 'email', 'minutes': 60},
                                            ]

                                            if minutes_before_now > 0:
                                                overrides_list.append({'method': 'popup', 'minutes': minutes_before_now})
                                                logger.info(f"Adding immediate debug reminder, scheduled {minutes_before_now} minutes before event.")
                                            else:
                                                logger.info("Event is in the past or starting now, skipping immediate debug reminder.")

                                            event_body = {
                                                'summary': task_desc,
                                                'description': f"Task detected by Kairo from session: {session_id}\n\nSource text: \"{text}\"",
                                                'start': {
                                                    'dateTime': start_dt.isoformat(),
                                                    'timeZone': tz_str,
                                                },
                                                'end': {
                                                    'dateTime': end_dt.isoformat(),
                                                    'timeZone': tz_str,
                                                },
                                                'reminders': {
                                                    'useDefault': False,
                                                    'overrides': overrides_list,
                                                },
                                            }

                                            event_id = await calendar_handler.create_event(event_body)

                                            if event_id:
                                                logger.info(f"Successfully created calendar event {event_id} for task {inserted_id}.")
                                                await self.mongo.update_task_with_calendar_id(inserted_id, event_id)
                                            else:
                                                logger.warning(f"Failed to create calendar event for task {inserted_id} (API call failed).")

                                        # Keep general exception handler for other unexpected issues
                                        except Exception as e:
                                             logger.error(f"Error processing or creating Google Calendar event for task {inserted_id}: {e}", exc_info=True)

                                    elif task_desc:
                                        logger.info(f"Task '{task_desc}' was detected but has no due date ('due_date_time' is {task_due_str}). Skipping calendar creation.")

                                elif not inserted_id:
                                     logger.error("Task was detected but failed to save to MongoDB. Skipping calendar creation.")
                                elif not calendar_handler:
                                     logger.info(f"Task {inserted_id} detected, but calendar handler is not available. Skipping calendar creation.")

                        # --- END OF TASK LOGIC ---
                        
                        full_session_transcript_parts.append(f"{segment['speaker']}: {text}")

                processed_audio_paths.append(preprocessed_path)
                current_session_offset += chunk_duration 

            if all_segments_for_pinecone:
                self.pinecone.upsert_records(records=all_segments_for_pinecone)
                logger.info(f"Upserted {len(all_segments_for_pinecone)} segments to Pinecone for session {session_id}.")
            else:
                logger.warning(f"No segments generated for Pinecone for session {session_id}.")
                
                
            # --- MEMORY GENERATION STEP ---
            if full_session_transcript_parts:
                logger.info(f"Generating memory for session {session_id}...")
                full_transcript_for_llm = "\n".join(full_session_transcript_parts)

                generated_memory = await self.memories_generator.generate_memory(user_id, full_transcript_for_llm)

                if generated_memory:
                    generated_memory['session_id'] = session_id
                    generated_memory['user_id'] = user_id
                    generated_memory['full_transcription'] = full_transcript_for_llm
                    generated_memory['duration_seconds'] = int(current_session_offset)
                    generated_memory['created_at'] = datetime.utcnow()

                    await self.mongo.save_memory(generated_memory)
                    logger.info(f"Successfully generated and saved memory for session {session_id}.")
                else:
                    logger.error(f"Failed to generate memory for session {session_id}.")
            else:
                logger.warning(f"No transcription parts to generate memory for session {session_id}.")
            # --- END MEMORY GENERATION ---

            if processed_audio_paths and drive_handler:
                # 4. Merge, Encrypt, and Upload to Google Drive
                key, drive_id = await asyncio.to_thread(
                    drive_handler.merge_and_encrypt_upload,
                    processed_audio_paths, session_id, user_id
                )
                
                if key and drive_id:
                    await self.mongo.update_session_status(session_id, 'archived', user_id, {'driveId': drive_id, 'encryptionKey': key, 'archiveDate': datetime.utcnow()})
                    logger.info(f"Session {session_id} archived to Google Drive with ID: {drive_id}")
                    await self.mongo.update_memory_with_drive_id(session_id, drive_id)
                else:
                    logger.error(f"Failed to archive session {session_id} to Google Drive.")
            elif not drive_handler:
                 logger.error(f"Google Drive handler not initialized. Skipping archive for session {session_id}.")
            else:
                logger.warning("Session contained no audio chunks with speech to process or archive.")

            await self.mongo.update_session_status(session_id, 'completed', user_id)
            logger.info(f"--- Successfully completed processing for session: {session_id} ---")
            
        except Exception as e:
            logger.error(f"Top-level processing failed for {session_id}: {e}", exc_info=True)
            await self.mongo.update_session_status(session_id, 'failed', user_id, {'error': str(e)})
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")

    async def run_pipeline(self):
        logger.info("Starting Kairo Pipeline Polling Loop...")
        while True:
            try:
                user_ids = await self.mongo.get_distinct_users()
                if not user_ids:
                    logger.info("No users found. Sleeping.")
                for user_id in user_ids:
                    logger.info(f"Checking for sessions for user: {user_id}")
                    pending_sessions = await self.tracker.get_pending_sessions(user_id)
                    if not pending_sessions:
                        logger.info(f"No new sessions for user {user_id}")
                        continue
                    
                    for session_id in pending_sessions:
                        await self.process_session(session_id, user_id)
                        
                interval = config.get('polling_interval', 20)
                logger.info(f"Polling loop finished. Sleeping for {interval} seconds.")
                await asyncio.sleep(interval)
            except Exception as e:
                logger.critical(f"A critical error occurred in the main pipeline loop: {e}", exc_info=True)
                await asyncio.sleep(config.get('critical_error_sleep_seconds', 60))