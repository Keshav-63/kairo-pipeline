from datetime import datetime
import os
import re
import sys
import yaml
import nest_asyncio
import json
import io
import asyncio
import threading
from dotenv import load_dotenv
import numpy as np
import soundfile as sf
nest_asyncio.apply()
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
from flask import Flask, request, Response, send_file
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
from bson import ObjectId
from bson.json_util import dumps
# NEW: Import WsgiToAsgi
from asgiref.wsgi import WsgiToAsgi
from pydub import AudioSegment
import shutil
import re

# Ensure the project root is in the Python path for correct module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from storage.mongodb_handler import SyncMongoDBHandler, AsyncMongoDBHandler
from storage.pinecone_handler import PineconeHandler
from storage.azure_blob import AzureBlobHandler
from utils.time_parser import parse_time_query
from utils.query_parser import parse_query_for_speaker_and_keywords
from config.logger import logger
from storage.google_drive import GoogleDriveHandler
from core.voice_recognition import VoiceRecognizer
# NEW: Import the Langchain LLM Handler
from utils.langchain_llm_handler import LangchainLLMHandler

load_dotenv()
if not os.getenv('GOOGLE_CLIENT_ID') or not os.getenv('GOOGLE_CLIENT_SECRET'):
    logger.critical("Missing required Google OAuth credentials in environment variables")
    exit(1)
    
app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

# --- Configuration Loading ---
try:
    config_path = os.path.join(project_root, 'config', 'settings.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logger.critical(f"Configuration file not found at {config_path}. Cannot start.")
    exit(1)
except Exception as e:
    logger.critical(f"Error loading configuration: {e}", exc_info=True)
    exit(1)

# --- Service Initialization ---
try:
    mongo_sync = SyncMongoDBHandler(config['mongodb']['uri'])
    mongo_async = AsyncMongoDBHandler(config['mongodb']['uri'])
    pinecone = PineconeHandler(config['pinecone']['api_key'], config['pinecone']['index_name'])
    
    # NEW: Initialize LangchainLLMHandler
    llm_handler = LangchainLLMHandler(config_path) 

    # drive = GoogleDriveHandler(config['google_drive']['service_account_json'])
    voice_recognizer = VoiceRecognizer()
    azure_handler = AzureBlobHandler(config['azure']['account_name'], config['azure']['account_key'], config['azure']['container'])
except Exception as e:
    logger.critical(f"Failed to initialize a core service: {e}", exc_info=True)
    exit(1)



# --- NEW CHAT HISTORY ENDPOINTS ---

@app.route('/chats', methods=['POST'])
async def create_chat():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        if not user_id:
            raise BadRequest("'user_id' is required.")
        
        session_id = await mongo_async.create_chat_session(user_id, "New Chat")
        return Response(response=json.dumps({"session_id": session_id}), status=201, mimetype='application/json')
    except BadRequest as e:
        return Response(response=json.dumps({"error": str(e)}), status=400, mimetype='application/json')
    except Exception as e:
        logger.critical(f"A critical error occurred in /chats endpoint: {e}", exc_info=True)
        return Response(response=json.dumps({"error": "An internal server error occurred."}), status=500, mimetype='application/json')

@app.route('/chats/user/<user_id>', methods=['GET'])
async def get_user_chats(user_id):
    try:
        sessions = await mongo_async.get_user_chat_sessions(user_id)
        return Response(response=dumps(sessions), status=200, mimetype='application/json')
    except Exception as e:
        logger.critical(f"A critical error occurred in /chats/user/{user_id} endpoint: {e}", exc_info=True)
        return Response(response=json.dumps({"error": "An internal server error occurred."}), status=500, mimetype='application/json')

@app.route('/chats/<session_id>', methods=['GET'])
async def get_chat_history(session_id):
    try:
        session = await mongo_async.get_chat_session(session_id)
        if not session:
            return Response(response=json.dumps({"error": "Chat session not found."}), status=404, mimetype='application/json')
        return Response(response=dumps(session), status=200, mimetype='application/json')
    except Exception as e:
        logger.critical(f"A critical error occurred in /chats/{session_id} endpoint: {e}", exc_info=True)
        return Response(response=json.dumps({"error": "An internal server error occurred."}), status=500, mimetype='application/json')




# --- Define a relevance threshold (adjust as needed) ---
RELEVANCE_THRESHOLD = 0.79


@app.route('/query', methods=['POST'])
async def query():
    try:
        # Step 1: Validate Content-Type header
        content_type = request.headers.get('Content-Type', '').lower()
        if not content_type.startswith('application/json'):
            logger.error(f"Unsupported Media Type: Expected 'application/json', got '{content_type}'")
            return Response(
                response=json.dumps({"error": "Unsupported Media Type: Please send data as 'application/json'."}),
                status=415,
                mimetype='application/json'
            )

        # Step 2: Parse JSON data
        data = request.get_json()
        if not data:
            logger.error("Failed to parse JSON data from request.")
            return Response(
                response=json.dumps({"error": "Invalid JSON data in request body."}),
                status=400,
                mimetype='application/json'
            )

        query_text = data.get('query')
        user_id = data.get('user_id')
        session_id = data.get('session_id')
        if not all([query_text, user_id, session_id]):
            logger.error(f"Missing required fields: 'query', 'user_id', or 'session_id' in {data}")
            return Response(
                response=json.dumps({"error": "Missing required fields: 'query', 'user_id', 'session_id'"}),
                status=400,
                mimetype='application/json'
            )

        logger.info(f"Received query from user {user_id} in session {session_id}: '{query_text[:200]}...'")

        # Step 3: Load chat history
        chat_session = await mongo_async.get_chat_session(session_id)
        chat_history = chat_session.get('history', []) if chat_session else []

        # Step 4: Get standalone question
        standalone_query = await llm_handler.get_standalone_question(user_id, query_text, chat_history)
        logger.debug(f"Standalone query: {standalone_query}")

        # Step 5: Parse query details
        start_ts, end_ts = parse_time_query(standalone_query)
        cleaned_query, speaker_filter, keywords = await parse_query_for_speaker_and_keywords(standalone_query, user_id, mongo_async)

        # Step 6: Determine if memory-related
        is_memory_query = bool(start_ts or speaker_filter or keywords or re.search(r'\b(what|discuss|said|talked|conversation|recall|remember|yesterday|last|today)\b', standalone_query.lower()))

        full_transcripts_context = [] # To store fetched full transcripts
        search_results_hits = []      # To store normalized hits from Pinecone
        context_text = ""             # For sending to the LLM

        if is_memory_query:
            # Step 7: Pinecone search with progressive relaxation
            # (Pinecone.semantic_search returns the normalized list directly now)
            search_results_hits = pinecone.semantic_search(
                query=standalone_query, user_id=user_id, top_k=15, # Fetch more initially for filtering
                start_timestamp=start_ts, end_timestamp=end_ts,
                speaker=speaker_filter, keywords=keywords
            )
            logger.info(f"Initial Pinecone search hits: {len(search_results_hits)}")

            if not search_results_hits:
                logger.warning("No initial hits; relaxing filters (remove speaker/keywords).")
                search_results_hits = pinecone.semantic_search(
                    query=standalone_query, user_id=user_id, top_k=12,
                    start_timestamp=start_ts, end_timestamp=end_ts
                )
            if not search_results_hits and (start_ts or end_ts):
                logger.warning("Still no hits; relaxing time filter.")
                search_results_hits = pinecone.semantic_search(query=standalone_query, user_id=user_id, top_k=20)
 
            # # --- START: MODIFIED LOGIC ---
            # # Step 8: Filter hits by score and fetch relevant full transcripts
            # if search_results_hits:
            #     # Filter hits based on the relevance score threshold
            #     relevant_hits = [hit for hit in search_results_hits if hit.get('_score', 0) >= RELEVANCE_THRESHOLD]
            #     logger.info(f"Filtered down to {len(relevant_hits)} relevant hits (score >= {RELEVANCE_THRESHOLD}).")

            #     if relevant_hits:
            #         # Extract unique session IDs ONLY from the relevant hits
            #         relevant_session_ids = list(set([hit.get('sessionId') for hit in relevant_hits if hit.get('sessionId')]))
            #         logger.debug(f"Fetching full transcripts for relevant session IDs: {relevant_session_ids}")

            #         if relevant_session_ids:
            #             # Fetch the full transcripts for these relevant sessions
            #             # Assuming get_transcripts_for_sessions returns a list of strings
            #             full_transcripts_context = await mongo_async.get_transcripts_for_sessions(relevant_session_ids)
            #             logger.info(f"Retrieved {len(full_transcripts_context)} relevant full transcripts from MongoDB.")
                        
            #             # Prepare context text for LLM using the fetched full transcripts
            #             context_text = "\n\n".join(full_transcripts_context)
            #         else:
            #             logger.warning("Relevant hits found, but no session IDs could be extracted.")
            #     else:
            #          logger.warning(f"No hits met the relevance threshold of {RELEVANCE_THRESHOLD}.")
            # # --- END: MODIFIED LOGIC ---
            
            # --- START: MODIFIED LOGIC (IMPROVED DATE ATTRIBUTION) ---
            if search_results_hits:
                # Filter hits based on the relevance score threshold
                relevant_hits = [hit for hit in search_results_hits if hit.get('_score', 0) >= RELEVANCE_THRESHOLD]
                logger.info(f"Filtered down to {len(relevant_hits)} relevant hits (score >= {RELEVANCE_THRESHOLD}).")

                if not relevant_hits:
                    logger.warning(f"No hits met the relevance threshold of {RELEVANCE_THRESHOLD}.")
                else:
                    # Extract unique session IDs from the relevant hits
                    relevant_session_ids = list({hit.get('sessionId') for hit in relevant_hits if hit.get('sessionId')})
                    logger.debug(f"Fetching full transcripts for relevant session IDs: {relevant_session_ids}")

                    if relevant_session_ids:
                        # Fetch the full transcripts for these relevant sessions
                        full_transcripts_context = await mongo_async.get_transcripts_for_sessions(relevant_session_ids)
                        logger.info(f"Retrieved {len(full_transcripts_context)} relevant full transcripts from MongoDB.")
                        logger.debug(f"FULL_transcripts_context ->:{full_transcripts_context}")
            
                        # --- Build context_text, including per-hit metadata (session + date) ---
                        # Map sessionId -> createdAt (take the max createdAt found for that session among relevant_hits)
                        session_created_map = {}
                        for hit in relevant_hits:
                            sid = hit.get('sessionId')
                            created = hit.get('createdAt')  # createdAt in your hits (float or int)
                            if not sid or created is None:
                                continue
                            # Use the latest createdAt for the session
                            if sid not in session_created_map or created > session_created_map[sid]:
                                session_created_map[sid] = int(created)

            # Choose a representative date for the LLM:
            # Prefer the top-most hit's createdAt (most semantically relevant),
            # otherwise the most recent among relevant_hits
                        representative_ts = None
                        try:
                            # top hit if exists
                            top_hit = max(relevant_hits, key=lambda h: h.get('_score', 0))
                            representative_ts = int(top_hit.get('createdAt')) if top_hit.get('createdAt') else None
                        except Exception:
                            representative_ts = None

                        if representative_ts is None:
                # fallback to most recent createdAt among relevant_hits
                            created_vals = [int(h.get('createdAt')) for h in relevant_hits if h.get('createdAt')]
                            if created_vals:
                                representative_ts = max(created_vals)

            # Build a rich context_text that includes session/date metadata — this helps the LLM.
            # Convert createdAt to readable local date string for each session
                        from datetime import datetime, timezone
                        contexts_with_meta = []
                        for ctx in full_transcripts_context:
                # try to extract sessionId from the transcript context string if possible,
                # otherwise default to listing text only.
                # We will also attach a date if we can find a session in session_created_map.
                # Here we assume get_transcripts_for_sessions returned contexts in same order as session_ids.
                # To be robust, we try to parse the sessionId prefix if present in your assembled transcripts.
                # Fallback: show context without explicit date.
                            contexts_with_meta.append(ctx)

            # Simpler approach: prepend metadata lines using session_created_map
            # We'll create a mapping of sessionId -> transcript text for clarity
            # (If your get_transcripts_for_sessions returns transcripts in the same order as session IDs,
            # you can pair them; else you might need a more reliable structure. For now we will include the metadata individually.)
                        context_text_parts = []
                        for sid, cts in session_created_map.items():
                            try:
                                # convert epoch to local date string in Asia/Kolkata
                                local_dt = datetime.fromtimestamp(cts)
                                # If tzinfo absent, local_dt is naive; treat as local wall-clock (avoid double-conversion downstream)
                                date_str = local_dt.strftime('%B %d, %Y')
                            except Exception:
                                date_str = "Date not available"
                                # find a transcript for this session from full_transcripts_context (simple substring match)
                            matching_transcript = next((t for t in full_transcripts_context if t["sessionId"] == sid), None)
                            if matching_transcript:
                                context_text_parts.append(f"[Session: {sid} | Date: {date_str}] {matching_transcript['text']}")
                            else:
                                context_text_parts.append(f"[Session: {sid} | Date: {date_str}] (transcript omitted)")


                            # If none matched (edge case), fallback to raw joined transcripts
                        if not context_text_parts:
                            context_text = "\n\n".join(full_transcripts_context)
                        else:
                            context_text = "\n\n".join(context_text_parts)

                            # Set start_ts_for_llm to the representative_ts (these are Pinecone-createdAt timestamps)
                        start_ts_for_llm = representative_ts

                    else:
                        logger.warning("Relevant hits found, but no session IDs could be extracted.")
            else:
                logger.warning("No initial hits; relaxing filters (remove speaker/keywords).")
# --- END: MODIFIED LOGIC (IMPROVED DATE ATTRIBUTION) ---


        # Step 9: LLM Synthesis
        if not context_text:
            logger.warning("No relevant context built. Using fallback message.")
            # Provide a more informative fallback message
            context_text = f"I couldn't find specific recordings matching your request with high confidence (threshold {RELEVANCE_THRESHOLD}). You might want to try rephrasing or broadening your search."
            
        logger.debug(f"FULL CONTEXT FOR LLM : -> {context_text}")
        llm_start_ts = locals().get('start_ts_for_llm') or start_ts or None
        answer = await llm_handler.get_answer_from_context(
            user_id,
            standalone_query,
            context_text,
            chat_history,
            start_ts=llm_start_ts
        )
        logger.info(f"LLM generated answer: {answer[:150]}...")

        # Step 10: Save to history and respond
        # Use only the relevant_hits (or all if none met threshold) for context IDs
        context_ids_to_save = [hit.get('_id', '') for hit in relevant_hits or search_results_hits]
        await mongo_async.save_query_history(user_id, query_text, answer, context_ids_to_save)

        await mongo_async.add_message_to_history(session_id, 'user', query_text)
        await mongo_async.add_message_to_history(session_id, 'assistant', answer)

        # Send back the filtered relevant hits as contexts/sources
        response = {
            "answer": answer,
            # Send the filtered hits back
            "contexts": relevant_hits if relevant_hits else search_results_hits,
            "sources": relevant_hits if relevant_hits else search_results_hits,
        }
        return Response(response=json.dumps(response), status=200, mimetype='application/json')

    except BadRequest as e:
        logger.error(f"Bad request: {e}")
        return Response(response=json.dumps({"error": str(e)}), status=400, mimetype='application/json')
    except Exception as e:
        logger.critical(f"Query error: {e}", exc_info=True)
        return Response(response=json.dumps({"error": "Internal server error."}), status=500, mimetype='application/json')
    
    
# # In app.py, update the /query route (async def query()):
# @app.route('/query', methods=['POST'])
# async def query():
#     try:
#         # Step 1: Validate Content-Type header
#         content_type = request.headers.get('Content-Type', '').lower()
#         if not content_type.startswith('application/json'):
#             logger.error(f"Unsupported Media Type: Expected 'application/json', got '{content_type}'")
#             return Response(
#                 response=json.dumps({"error": "Unsupported Media Type: Please send data as 'application/json'."}),
#                 status=415,
#                 mimetype='application/json'
#             )

#         # Step 2: Parse JSON data
#         data = request.get_json()
#         if not data:
#             logger.error("Failed to parse JSON data from request.")
#             return Response(
#                 response=json.dumps({"error": "Invalid JSON data in request body."}),
#                 status=400,
#                 mimetype='application/json'
#             )

#         query_text = data.get('query')
#         user_id = data.get('user_id')
#         session_id = data.get('session_id')
#         if not all([query_text, user_id, session_id]):
#             logger.error(f"Missing required fields: 'query', 'user_id', or 'session_id' in {data}")
#             return Response(
#                 response=json.dumps({"error": "Missing required fields: 'query', 'user_id', 'session_id'"}),
#                 status=400,
#                 mimetype='application/json'
#             )

#         logger.info(f"Received query from user {user_id} in session {session_id}: '{query_text[:200]}...'")

#             # Step 3: Load chat history for context/follow-ups
#         chat_session = await mongo_async.get_chat_session(session_id)
#         chat_history = chat_session.get('history', []) if chat_session else []

#         # Step 4: Get standalone question (contextualized with history)
#         standalone_query = await llm_handler.get_standalone_question(user_id, query_text, chat_history)
#         logger.debug(f"Standalone query: {standalone_query}")

#         # Step 5: Parse for time, speaker, keywords
#         start_ts, end_ts = parse_time_query(standalone_query)
#         cleaned_query, speaker_filter, keywords = await parse_query_for_speaker_and_keywords(standalone_query, user_id, mongo_async)

#         # Step 6: Determine if memory-related
#         is_memory_query = bool(start_ts or speaker_filter or keywords or re.search(r'\b(what|discuss|said|talked|conversation|recall|remember|yesterday|last|today)\b', standalone_query.lower()))

#         contexts = []
#         search_results = []

#         if is_memory_query:
#             # Step 7: Pinecone search with progressive relaxation
#             search_results = pinecone.semantic_search(
#                 query=standalone_query, user_id=user_id, top_k=10,
#                 start_timestamp=start_ts, end_timestamp=end_ts,
#                 speaker=speaker_filter, keywords=keywords
#             )
#             logger.info(f"Initial Pinecone search hits: {len(search_results)}")

#             if not search_results:
#                 logger.warning("No initial hits; relaxing filters (remove speaker/keywords).")
#                 search_results = pinecone.semantic_search(
#                     query=standalone_query, user_id=user_id, top_k=10,
#                     start_timestamp=start_ts, end_timestamp=end_ts
#                 )
#             if not search_results and (start_ts or end_ts):
#                 logger.warning("Still no hits; relaxing time filter.")
#                 search_results = pinecone.semantic_search(query=standalone_query, user_id=user_id, top_k=10)

#             # Step 8: Get transcripts if hits
#             if search_results:
#                 session_ids = list(set([result.get('sessionId') for result in search_results]))
#                 contexts = await mongo_async.get_transcripts_for_sessions(session_ids)
#                 logger.info(f"Retrieved {len(contexts)} transcripts from MongoDB.")
#                 logger.info(f"Retrieved FULL COONTEXT {contexts} transcripts from MongoDB.")

#         # Step 9: LLM Synthesis
#         context_text = "\n\n".join(contexts) if contexts else "No recorded contexts found. Possible reasons: No pendant activity during the specified time/speaker, or filters were too strict. Suggest checking pendant logs or enrolling more voices."
#         answer = await llm_handler.get_answer_from_context(user_id, standalone_query, context_text, chat_history)
#         logger.info(f"LLM generated answer: {answer}...")

#         # Step 10: Save to history and respond
#         context_ids = [result.get('_id', '') for result in search_results]
#         await mongo_async.save_query_history(user_id, query_text, answer, context_ids)

#         await mongo_async.add_message_to_history(session_id, 'user', query_text)
#         await mongo_async.add_message_to_history(session_id, 'assistant', answer)

#         response = {
#             "answer": answer,
#             "contexts": contexts,
#             "sources": search_results,
#         }
#         return Response(response=json.dumps(response), status=200, mimetype='application/json')

#     except BadRequest as e:
#         logger.error(f"Bad request: {e}")
#         return Response(response=json.dumps({"error": str(e)}), status=400, mimetype='application/json')
#     except Exception as e:
#         logger.critical(f"Query error: {e}", exc_info=True)
#         return Response(response=json.dumps({"error": "Internal error."}), status=500, mimetype='application/json')



# --- CORRECTED AUDIO STREAMING ENDPOINT ---
# @app.route('/audio/<session_id>', methods=['GET'])
# def get_audio(session_id):
#     try:
#         user_id = request.args.get('user_id')
#         if not user_id:
#             raise BadRequest("'user_id' is a required query parameter.")

#         logger.info(f"Received audio request for session '{session_id}' from user '{user_id}'")
        
#         user_credentials = mongo_sync.get_user_credentials(user_id)
#         if not user_credentials:
#             return Response(response=json.dumps({"error": "Could not find credentials for user."}), status=404, mimetype='application/json')
        
#         drive_handler = GoogleDriveHandler(user_credentials)
        
#         session_details = mongo_sync.get_session_details(session_id, user_id)
#         if not session_details or 'encryptionKey' not in session_details:
#             # We don't need driveId from the DB anymore, just the encryption key
#             return Response(response=json.dumps({"error": "Session details or encryption key not found."}), status=404, mimetype='application/json')
        
#         encryption_key = session_details['encryptionKey']

#         # --- START OF FIX ---
#         # Construct the folder and file names based on the corrected structure
#         folder_name = f"Kairo_{user_id}"
#         file_name = f"{session_id}.encrypted.wav"
#         # --- END OF FIX ---
        
#         drive_id = drive_handler.find_file_in_folder(folder_name, file_name)
#         if not drive_id:
#             logger.warning(f"File '{file_name}' not found in folder '{folder_name}' for user '{user_id}'.")
#             return Response(response=json.dumps({"error": "Audio file not found on Google Drive."}), status=404, mimetype='application/json')

#         decrypted_audio = drive_handler.download_and_decrypt_audio(drive_id, encryption_key)
#         if not decrypted_audio:
#             return Response(response=json.dumps({"error": "Failed to retrieve or decrypt audio."}), status=500, mimetype='application/json')
            
#         return send_file(
#             io.BytesIO(decrypted_audio),
#             mimetype='audio/wav',
#             as_attachment=False
#         )
        
#     except BadRequest as e:
#         logger.error(f"Bad request to /audio endpoint: {e}")
#         return Response(response=json.dumps({"error": str(e)}), status=400, mimetype='application/json')
#     except Exception as e:
#         logger.critical(f"A critical error occurred in the /audio endpoint: {e}", exc_info=True)
#         return Response(response=json.dumps({"error": "An internal server error occurred."}), status=500, mimetype='application/json')
@app.route('/audio/<session_id>', methods=['GET'])
async def get_audio(session_id):
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            raise BadRequest("'user_id' is a required query parameter.")

        logger.info(f"Received audio request for session '{session_id}' from user '{user_id}'")
        
        # Get user credentials
        user_credentials = mongo_sync.get_user_credentials(user_id)
        if not user_credentials:
            return Response(
                response=json.dumps({"error": "Could not find credentials for user."}), 
                status=404, 
                mimetype='application/json'
            )
        
        # Initialize drive handler with proper credentials and user_id
        drive_handler = GoogleDriveHandler(
            user_credentials_dict=user_credentials,
            mongo_handler=mongo_async,
            user_id=user_id
        )
        
        # Initialize the service before using it
        await drive_handler.initialize_service()
        
        session_details = mongo_sync.get_session_details(session_id, user_id)
        if not session_details or 'encryptionKey' not in session_details:
            return Response(
                response=json.dumps({"error": "Session details or encryption key not found."}), 
                status=404, 
                mimetype='application/json'
            )
        
        encryption_key = session_details['encryptionKey']
        folder_name = f"Kairo_{user_id}"
        file_name = f"{session_id}.encrypted.wav"
        
        drive_id = drive_handler.find_file_in_folder(folder_name, file_name)
        if not drive_id:
            logger.warning(f"File '{file_name}' not found in folder '{folder_name}' for user '{user_id}'.")
            return Response(
                response=json.dumps({"error": "Audio file not found on Google Drive."}), 
                status=404, 
                mimetype='application/json'
            )

        decrypted_audio = drive_handler.download_and_decrypt_audio(drive_id, encryption_key)
        if not decrypted_audio:
            return Response(
                response=json.dumps({"error": "Failed to retrieve or decrypt audio."}), 
                status=500, 
                mimetype='application/json'
            )
            
        return send_file(
            io.BytesIO(decrypted_audio),
            mimetype='audio/wav',
            as_attachment=False
        )
        
    except BadRequest as e:
        logger.error(f"Bad request to /audio endpoint: {e}")
        return Response(
            response=json.dumps({"error": str(e)}), 
            status=400, 
            mimetype='application/json'
        )
    except Exception as e:
        logger.critical(f"A critical error occurred in the /audio endpoint: {e}", exc_info=True)
        return Response(
            response=json.dumps({"error": "An internal server error occurred."}), 
            status=500, 
            mimetype='application/json'
        )


@app.route('/enroll', methods=['POST'])
def enroll_voice():
    temp_dir = None
    try:
        data = request.form
        user_id = data.get('user_id')
        person_name = data.get('person_name').strip().lower()
        relationship = data.get('relationship').strip().lower()
        logger.debug(f"Enrollment data received: user_id={user_id}, person_name={person_name}, relationship={relationship}")
        
        if not all([user_id, person_name, relationship]):
            raise BadRequest("'user_id', 'person_name', and 'relationship' are required.")

        if 'voice_sample' not in request.files:
            raise BadRequest("No voice sample file found in the request.")
        
        voice_sample = request.files['voice_sample']
        
        temp_dir = "temp_enrollment"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save and convert audio
        original_temp_path = os.path.join(temp_dir, voice_sample.filename)
        voice_sample.save(original_temp_path)

        try:
            audio = AudioSegment.from_file(original_temp_path)
            base_filename = os.path.splitext(voice_sample.filename)[0]
            temp_path = os.path.join(temp_dir, f"{base_filename}.wav")
            audio.export(temp_path, format="wav")
            logger.info(f"Successfully converted uploaded audio to WAV format at {temp_path}")
        except Exception as e:
            logger.error(f"Failed to convert audio file: {e}", exc_info=True)
            raise Exception("Could not process the uploaded audio file format.")
        finally:
            if os.path.exists(original_temp_path):
                os.remove(original_temp_path)

        # Data augmentation
        augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
        ])

        # Generate embeddings
        embeddings = []
        original_embedding = voice_recognizer.get_embedding(temp_path)
        if original_embedding is not None:
            embeddings.append(original_embedding)

        # Create augmented samples
        audio, sr = sf.read(temp_path)
        for i in range(5):
            augmented_audio = augment(samples=audio, sample_rate=sr)
            augmented_path = os.path.join(temp_dir, f"aug_{i}_{base_filename}.wav")
            sf.write(augmented_path, augmented_audio, sr)
            aug_embedding = voice_recognizer.get_embedding(augmented_path)
            if aug_embedding is not None:
                embeddings.append(aug_embedding)

        if not embeddings:
            raise Exception("Failed to generate any embeddings for the voice sample.")

        # Average embeddings
        embedding = np.mean(embeddings, axis=0)
        
        with open(temp_path, 'rb') as f:
            audio_data = f.read()

        sample_url = azure_handler.upload_voice_sample_sync(user_id, person_name, audio_data)
        if not sample_url:
            raise Exception("Failed to upload voice sample to Azure.")
        
        mongo_sync.save_enrollment(user_id, person_name, relationship, embedding, sample_url)
        logger.info(f"Saved enrollment for {person_name} to MongoDB.")

        return Response(response=json.dumps({"message": f"Voice enrollment for {person_name} is being processed."}), status=202, mimetype='application/json')

    except BadRequest as e:
        logger.error(f"Bad request to /enroll endpoint: {e}")
        return Response(response=json.dumps({"error": str(e)}), status=400, mimetype='application/json')
    except Exception as e:
        logger.critical(f"A critical error occurred in the /enroll endpoint: {e}", exc_info=True)
        return Response(response=json.dumps({"error": "An internal server error occurred."}), status=500, mimetype='application/json')
    finally:
        # Clean up ALL temporary files and directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up temp directory: {cleanup_error}")




@app.route('/chats/<session_id>/title', methods=['POST'])
async def generate_and_update_title(session_id):
    try:
        chat_session = await mongo_async.get_chat_session(session_id)
        if not chat_session:
            return Response(response=json.dumps({"error": "Chat session not found."}), status=404, mimetype='application/json')

        history = chat_session.get('history', [])
        
        # FIX: Check for at least one user and one assistant message
        has_user_message = any(msg.get('role') == 'user' for msg in history)
        has_assistant_message = any(msg.get('role') == 'assistant' for msg in history)

        if has_user_message and has_assistant_message:
            new_title = await llm_handler.generate_chat_title(history)
            if new_title and new_title != chat_session.get("title"):
                await mongo_async.update_chat_title(session_id, new_title)
                return Response(response=json.dumps({"message": "Title updated successfully.", "title": new_title}), status=200, mimetype='application/json')
        
        return Response(response=json.dumps({"message": "Not enough conversation history to generate a title."}), status=200, mimetype='application/json')

    except Exception as e:
        logger.critical(f"A critical error occurred in title generation endpoint: {e}", exc_info=True)
        return Response(response=json.dumps({"error": "An internal server error occurred."}), status=500, mimetype='application/json')
    

# --- ENDPOINT TO LIST RECORDINGS ---
@app.route('/recordings/<user_id>', methods=['GET'])
def get_user_recordings(user_id):
    try:
        logger.info(f"Received request for recordings list for user '{user_id}'")
        
        # Get sessions sorted by sessionId in descending order (most recent first)
        sessions = mongo_sync.get_all_user_sessions(user_id, sort_order=-1)
        
        if not sessions:
            logger.warning(f"No sessions found for user {user_id}")
            return Response(response=dumps([]), status=200, mimetype='application/json')
            
        recordings = []
        for session in sessions:
            if session.get('sessionId'):
                session_id = session.get('sessionId')
                details = session.get('details', {})
                
                # Extract timestamp from session ID
                timestamp = session_id.split('_')[-1] if '_' in session_id else None
                try:
                    date = datetime.fromtimestamp(int(timestamp)) if timestamp else datetime.utcnow()
                except ValueError:
                    date = datetime.utcnow()
                
                recordings.append({
                    "id": str(session['_id']),
                    "sessionId": session_id,
                    "title": details.get('title', session_id),
                    "durationSeconds": details.get('duration_seconds', 0),
                    "category": "personal",
                    "quality": "high",
                    "transcript": details.get('transcript', "Use Kairo AI to get the transcript."),
                    "waveform": [_ for _ in range(50)],
                    "date": date.isoformat(),
                    "timestamp": int(timestamp) if timestamp else int(date.timestamp())
                })
        
        # Sort recordings by timestamp in descending order
        recordings.sort(key=lambda x: x['timestamp'], reverse=True)
        
        logger.info(f"Returning {len(recordings)} recordings for user {user_id}, sorted by date (newest first)")
        return Response(response=dumps(recordings), status=200, mimetype='application/json')
        
    except Exception as e:
        logger.critical(f"A critical error occurred in the /recordings endpoint: {e}", exc_info=True)
        return Response(
            response=dumps({"error": "An internal server error occurred."}), 
            status=500, 
            mimetype='application/json'
        )


# --- NEW: MEMORIES API ENDPOINTS ---
@app.route('/memories/user/<user_id>', methods=['GET'])
async def get_user_memories(user_id):
    """
    Retrieves a list of all memories for a specific user.
    """
    try:
        logger.info(f"Request received for all memories for user: {user_id}")
        # You will need to implement 'get_memories_for_user' in your async mongodb_handler
        memories = await mongo_async.get_memories_for_user(user_id)
        if not memories:
            logger.debug(f"empty response yrr {memories}")
            return Response(response=json.dumps([]), status=200, mimetype='application/json')

        # Convert ObjectId to string for JSON serialization
        for memory in memories:
            if '_id' in memory:
                memory['_id'] = str(memory['_id'])
        logger.debug(f"meories response {memories}")
        return Response(response=dumps(memories), status=200, mimetype='application/json')
    except Exception as e:
        logger.critical(f"A critical error occurred in /memories/user/{user_id}: {e}", exc_info=True)
        return Response(response=json.dumps({"error": "An internal server error occurred."}), status=500, mimetype='application/json')


@app.route('/memories/<session_id>', methods=['GET'])
async def get_memory_by_session_id(session_id):
    """
    Retrieves a single memory by its session_id.
    """
    try:
        logger.info(f"Request received for memory with session_id: {session_id}")
        # You will need to implement 'get_memory' in your async mongodb_handler
        memory = await mongo_async.get_memory(session_id)
        if not memory:
            return Response(response=json.dumps({"error": "Memory not found."}), status=404, mimetype='application/json')

        # Convert ObjectId to string for JSON serialization
        if '_id' in memory:
            memory['_id'] = str(memory['_id'])

        return Response(response=dumps(memory), status=200, mimetype='application/json')
    except Exception as e:
        logger.critical(f"A critical error occurred in /memories/{session_id}: {e}", exc_info=True)
        return Response(response=json.dumps({"error": "An internal server error occurred."}), status=500, mimetype='application/json')

# --- END NEW ENDPOINTS ---
# In api/app.py, add these new endpoints

# --- NEW BEHAVIOR ANALYTICS ENDPOINTS ---

@app.route('/analytics/user/<user_id>/summary', methods=['GET'])
async def get_user_analytics_summary(user_id):
    """
    Retrieves a summary of daily analytics reports for a specific user.
    """
    try:
        logger.info(f"Request received for analytics summary for user: {user_id}")
        summary = await mongo_async.get_daily_analytics_summary(user_id)
        
        # Convert datetime objects to string for JSON serialization
        for entry in summary:
            if 'date' in entry and isinstance(entry['date'], datetime):
                entry['date'] = entry['date'].strftime('%Y-%m-%d')
            if 'updatedAt' in entry and isinstance(entry['updatedAt'], datetime):
                entry['updatedAt'] = entry['updatedAt'].isoformat()
                
        return Response(response=json.dumps(summary), status=200, mimetype='application/json')
    except Exception as e:
        logger.critical(f"A critical error occurred in /analytics/user/{user_id}/summary: {e}", exc_info=True)
        return Response(response=json.dumps({"error": "An internal server error occurred."}), status=500, mimetype='application/json')

@app.route('/analytics/user/<user_id>/<date_str>', methods=['GET'])
async def get_user_daily_analytics(user_id, date_str):
    """
    Retrieves detailed daily analytics for a specific user and date.
    Date_str format: YYYY-MM-DD
    """
    try:
        logger.info(f"Request received for daily analytics for user: {user_id} on date: {date_str}")
        daily_report = await mongo_async.get_daily_analytics_by_date(user_id, date_str)

        if not daily_report:
            return Response(response=json.dumps({"error": "Daily analytics report not found for this date."}), status=404, mimetype='application/json')

        # Convert datetime objects to string for JSON serialization
        if 'date' in daily_report and isinstance(daily_report['date'], datetime):
            daily_report['date'] = daily_report['date'].strftime('%Y-%m-%d')
        if 'updatedAt' in daily_report and isinstance(daily_report['updatedAt'], datetime):
            daily_report['updatedAt'] = daily_report['updatedAt'].isoformat()

        return Response(response=json.dumps(daily_report), status=200, mimetype='application/json')
    except BadRequest as e:
        logger.error(f"Bad request to /analytics/user/{user_id}/{date_str}: {e}")
        return Response(response=json.dumps({"error": str(e)}), status=400, mimetype='application/json')
    except Exception as e:
        logger.critical(f"A critical error occurred in /analytics/user/{user_id}/{date_str}: {e}", exc_info=True)
        return Response(response=json.dumps({"error": "An internal server error occurred."}), status=500, mimetype='application/json')
    
    
    
    
    
    
# --- NEW TASK MANAGEMENT ENDPOINTS ---

@app.route('/tasks/user/<user_id>', methods=['GET'])
async def get_user_tasks(user_id):
    """
    Retrieves all tasks for a specific user, optionally filtering by status.
    """
    try:
        status = request.args.get('status', 'pending') # Default to pending tasks
        logger.info(f"Request received for tasks for user: {user_id} with status: {status}")
        
        # You will need to add this method to your mongodb_handler
        tasks = await mongo_async.get_tasks_by_status(user_id, status)
        
        return Response(response=dumps(tasks), status=200, mimetype='application/json')
    except Exception as e:
        logger.critical(f"A critical error occurred in /tasks/user/{user_id}: {e}", exc_info=True)
        return Response(response=json.dumps({"error": "An internal server error occurred."}), status=500, mimetype='application/json')

@app.route('/tasks/<task_id>/complete', methods=['POST'])
async def complete_task(task_id):
    """
    Marks a specific task as 'complete'.
    """
    try:
        logger.info(f"Request received to mark task as complete: {task_id}")
        
        # You will need to add this method to your mongodb_handler
        result = await mongo_async.update_task_status(task_id, 'complete')

        if result.modified_count > 0:
            return Response(response=json.dumps({"message": "Task marked as complete."}), status=200, mimetype='application/json')
        else:
            return Response(response=json.dumps({"error": "Task not found or already complete."}), status=404, mimetype='application/json')
    except Exception as e:
        logger.critical(f"A critical error occurred in /tasks/{task_id}/complete: {e}", exc_info=True)
        return Response(response=json.dumps({"error": "An internal server error occurred."}), status=500, mimetype='application/json')

# NEW: Wrap your Flask app instance with WsgiToAsgi
asgi_app = WsgiToAsgi(app)

if __name__ == "__main__":
    # If running with 'python app.py' directly, keep the Flask run method
    # However, for ASGI features, 'uvicorn' or 'gunicorn -k uvicorn.workers.UvicornWorker' is preferred
    app.run(host="0.0.0.0", port=8000, debug=True)



