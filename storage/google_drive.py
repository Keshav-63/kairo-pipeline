# from googleapiclient.discovery import build
# from google.oauth2.credentials import Credentials
# from google.auth.transport.requests import Request as GoogleAuthRequest
# from google.auth.exceptions import RefreshError
# from cryptography.fernet import Fernet
# import io
# from config.logger import logger
# import os
# from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

# class GoogleDriveHandler:
#     def __init__(self, user_credentials, mongo_handler=None, user_id=None):
#         """
#         Initializes the GoogleDriveHandler with user-specific OAuth 2.0 credentials.
        
#         IMPROVED: Added automatic token refresh with persistence.
        
#         Args:
#             user_credentials: Dict with token, refresh_token, token_uri, client_id, client_secret, scopes
#             mongo_handler: Optional MongoDB handler to update tokens after refresh
#             user_id: Optional user ID for token updates
#         """
#         try:
#             logger.debug(f"Initializing GoogleDriveHandler with credential keys: {list(user_credentials.keys())}")
            
#             # CRITICAL: Validate all required fields are present
#             required_fields = ["token", "refresh_token", "token_uri", "client_id", "client_secret"]
#             missing = [f for f in required_fields if f not in user_credentials or not user_credentials[f]]
            
#             if missing:
#                 raise ValueError(
#                     f"Incomplete OAuth2 credentials. Missing fields: {missing}. "
#                     f"User must re-authenticate to obtain valid refresh_token."
#                 )
            
#             # Create credentials object
#             creds = Credentials(**user_credentials)
            
#             # Store for potential token updates
#             self.mongo_handler = mongo_handler
#             self.user_id = user_id
            
#             # IMPROVED: Refresh token if expired, with better error handling
#             if creds.expired and creds.refresh_token:
#                 logger.info("Access token expired. Attempting to refresh...")
#                 try:
#                     creds.refresh(GoogleAuthRequest())
#                     logger.info("Successfully refreshed access token.")
                    
#                     # CRITICAL: Save new tokens back to database
#                     if self.mongo_handler and self.user_id:
#                         import asyncio
#                         loop = asyncio.get_event_loop()
#                         if loop.is_running():
#                             # If in async context, schedule update
#                             asyncio.create_task(
#                                 self.mongo_handler.update_user_tokens(
#                                     self.user_id,
#                                     creds.token,
#                                     creds.refresh_token
#                                 )
#                             )
#                         else:
#                             # Sync context
#                             loop.run_until_complete(
#                                 self.mongo_handler.update_user_tokens(
#                                     self.user_id,
#                                     creds.token,
#                                     creds.refresh_token
#                                 )
#                             )
                    
#                 except RefreshError as e:
#                     logger.error(
#                         f"Failed to refresh access token: {e}. "
#                         f"Possible causes: "
#                         f"1) Refresh token expired (>6 months unused), "
#                         f"2) Refresh token revoked by user, "
#                         f"3) OAuth app in 'Testing' mode (tokens expire after 7 days). "
#                         f"User must re-authenticate."
#                     )
#                     raise
            
#             elif not creds.valid:
#                 logger.warning(
#                     "Credentials are not valid and cannot be refreshed. "
#                     "User needs to re-authenticate."
#                 )
#                 raise ValueError("Invalid credentials. User must re-authenticate.")
            
#             # Build Drive service
#             self.service = build("drive", "v3", credentials=creds)
#             logger.info("GoogleDriveHandler initialized successfully.")
            
#         except Exception as e:
#             logger.error(f"Failed to initialize GoogleDriveHandler: {e}", exc_info=True)
#             raise

#     def get_user_folder(self, user_id):
#         """Gets or creates user folder in Drive."""
#         folder_name = f'Kairo_{user_id}'
#         logger.debug(f"Searching for Google Drive folder with name: '{folder_name}'")
        
#         try:
#             query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
#             results = self.service.files().list(q=query, fields="files(id, name)").execute()
            
#             if results.get('files'):
#                 folder_id = results['files'][0]['id']
#                 logger.debug(f"Found existing folder for user {user_id} with ID: {folder_id}")
#                 return folder_id
            
#             logger.info(f"No folder found for user {user_id}. Creating a new one...")
#             file_metadata = {
#                 'name': folder_name,
#                 'mimeType': 'application/vnd.google-apps.folder'
#             }
#             folder = self.service.files().create(body=file_metadata, fields='id').execute()
#             folder_id = folder.get('id')
#             logger.info(f"Created new folder for user {user_id} with ID: {folder_id}")
#             return folder_id
            
#         except Exception as e:
#             logger.error(f"Failed to get/create user folder: {e}", exc_info=True)
#             raise

#     def merge_and_encrypt_upload(self, chunk_paths, session_id, user_id):
#         """Merges, encrypts, and uploads audio to Drive."""
#         try:
#             logger.debug(f"Merging {len(chunk_paths)} audio chunks for session {session_id}...")
#             merged_audio = b''
#             for path in chunk_paths:
#                 with open(path, 'rb') as f:
#                     merged_audio += f.read()
            
#             logger.debug(f"Merged audio size: {len(merged_audio)} bytes.")
            
#             logger.debug("Generating encryption key and encrypting merged audio...")
#             key = Fernet.generate_key()
#             fernet = Fernet(key)
#             encrypted = fernet.encrypt(merged_audio)
#             logger.debug("Encryption complete.")
            
#             folder_id = self.get_user_folder(user_id)
#             if not folder_id:
#                 raise Exception("Failed to get or create a user folder in Google Drive.")
            
#             file_name = f'{session_id}.encrypted.wav'
#             file_metadata = {
#                 'name': file_name,
#                 'parents': [folder_id]
#             }
#             media = MediaIoBaseUpload(io.BytesIO(encrypted), mimetype='audio/wav', resumable=True)
            
#             logger.info(f"Uploading '{file_name}' to Google Drive folder ID: {folder_id}...")
#             file = self.service.files().create(
#                 body=file_metadata,
#                 media_body=media,
#                 fields='id'
#             ).execute()
            
#             file_id = file.get('id')
#             logger.info(f"Successfully uploaded encrypted session {session_id} to Drive. File ID: {file_id}")
            
#             return key.decode(), file_id
            
#         except Exception as e:
#             logger.error(f"Drive upload failed for session {session_id}: {e}", exc_info=True)
#             return None, None

#     def download_and_decrypt_audio(self, file_id, encryption_key):
#         """Downloads and decrypts audio from Drive."""
#         try:
#             logger.info(f"Starting download for Google Drive file ID: {file_id}")
#             request = self.service.files().get_media(fileId=file_id)
#             fh = io.BytesIO()
#             downloader = MediaIoBaseDownload(fh, request)
            
#             done = False
#             while not done:
#                 status, done = downloader.next_chunk()
#                 if status:
#                     logger.debug(f"Download progress: {int(status.progress() * 100)}%")
            
#             logger.info("Download complete. Decrypting file...")
#             fh.seek(0)
#             encrypted_data = fh.read()
            
#             fernet = Fernet(encryption_key.encode())
#             decrypted_data = fernet.decrypt(encrypted_data)
            
#             logger.info(f"File {file_id} decrypted successfully. Size: {len(decrypted_data)} bytes.")
#             return decrypted_data
            
#         except Exception as e:
#             logger.error(f"Failed to download and decrypt file {file_id}: {e}", exc_info=True)
#             return None

#     def find_file_in_folder(self, folder_name: str, file_name: str) -> str:
#         """Finds a file by name within a specific folder."""
#         try:
#             # First, find the folder ID
#             folder_query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
#             folder_response = self.service.files().list(
#                 q=folder_query,
#                 spaces='drive',
#                 fields='files(id, name)'
#             ).execute()
            
#             folders = folder_response.get('files', [])
#             if not folders:
#                 logger.warning(f"Could not find the parent folder '{folder_name}' on Google Drive.")
#                 return None
            
#             folder_id = folders[0]['id']
            
#             # Search for file in folder
#             file_query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
#             file_response = self.service.files().list(
#                 q=file_query,
#                 spaces='drive',
#                 fields='files(id, name)'
#             ).execute()
            
#             files = file_response.get('files', [])
#             if files:
#                 file_id = files[0]['id']
#                 logger.info(f"Found file '{file_name}' with ID '{file_id}' inside folder '{folder_name}'.")
#                 return file_id
#             else:
#                 logger.warning(f"Could not find the file '{file_name}' inside the folder '{folder_name}'.")
#                 return None
                
#         except Exception as e:
#             logger.error(
#                 f"An error occurred while searching for file '{file_name}' in folder '{folder_name}': {e}",
#                 exc_info=True
#             )
#             return None




# storage/google_drive.py

import asyncio
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.auth.exceptions import RefreshError
from cryptography.fernet import Fernet
import io
from config.logger import logger
import os
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

class GoogleDriveHandler:
    def __init__(self, user_credentials_dict, mongo_handler=None, user_id=None):
        """
        Initializes the handler with raw credentials.
        Does NOT build the service yet.
        """
        self.user_credentials_dict = user_credentials_dict
        self.mongo_handler = mongo_handler
        self.user_id = user_id
        self.service = None # Will be built by initialize_service
        self.creds = None   # Will be loaded by initialize_service
        logger.debug(f"GoogleDriveHandler instantiated for user {user_id}")

    async def initialize_service(self):
        """
        Asynchronously loads credentials, refreshes them if necessary,
        saves the new token, and builds the Google Drive service.
        """
        try:
            required_fields = ["token", "refresh_token", "token_uri", "client_id", "client_secret"]
            missing = [f for f in required_fields if f not in self.user_credentials_dict or not self.user_credentials_dict[f]]
            
            if missing:
                raise ValueError(
                    f"Incomplete OAuth2 credentials for user {self.user_id}. Missing fields: {missing}. "
                    f"User must re-authenticate."
                )
            
            self.creds = Credentials.from_authorized_user_info(self.user_credentials_dict)
            
        except Exception as e:
            logger.error(f"Failed to load credentials for user {self.user_id}: {e}")
            raise ValueError(f"Invalid credentials format: {e}")

        if self.creds.expired and self.creds.refresh_token:
            logger.info(f"Google Drive token for user {self.user_id} has expired. Refreshing...")
            try:
                await asyncio.to_thread(self.creds.refresh, GoogleAuthRequest())
                
                if self.mongo_handler and self.user_id:
                    new_creds_dict = {
                        'token': self.creds.token,
                        'refresh_token': self.creds.refresh_token,
                        'token_uri': self.creds.token_uri,
                        'client_id': self.creds.client_id,
                        'client_secret': self.creds.client_secret,
                        'scopes': self.creds.scopes,
                        'expiry': self.creds.expiry.isoformat()
                    }
                    await self.mongo_handler.update_user_credentials(self.user_id, new_creds_dict)
                    logger.info(f"Successfully refreshed and saved Drive token for user {self.user_id}")
                else:
                    logger.warning(f"Drive token for {self.user_id} refreshed, but mongo_handler or user_id is missing. Cannot save.")
            
            except RefreshError as e:
                logger.error(f"Failed to refresh Drive token for user {self.user_id}: {e}", exc_info=True)
                raise IOError(f"Token refresh failed: {e}. User may need to re-authenticate.")
            except Exception as e:
                logger.error(f"An unexpected error occurred during Drive token refresh for {self.user_id}: {e}", exc_info=True)
                raise
        
        elif not self.creds.valid:
             logger.warning(f"Drive credentials for {self.user_id} are not valid and no refresh token is available.")
             raise ValueError("Invalid credentials and no refresh token.")

        try:
            self.service = build("drive", "v3", credentials=self.creds)
            logger.info(f"Google Drive service initialized for user {self.user_id}")
        except Exception as e:
            logger.error(f"Failed to build Google Drive service: {e}", exc_info=True)
            raise

    def _check_service(self):
        """Internal check to ensure service is initialized."""
        if not self.service:
            logger.error(f"Google Drive service for user {self.user_id} is not initialized. Call await initialize_service() first.")
            raise RuntimeError("Service not initialized")

    def get_user_folder(self, user_id):
        """Gets or creates user folder in Drive. (Blocking)"""
        self._check_service()
        folder_name = f'Kairo_{user_id}'
        logger.debug(f"Searching for Google Drive folder with name: '{folder_name}'")
        
        try:
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.service.files().list(q=query, fields="files(id, name)").execute()
            
            if results.get('files'):
                folder_id = results['files'][0]['id']
                logger.debug(f"Found existing folder for user {user_id} with ID: {folder_id}")
                return folder_id
            
            logger.info(f"No folder found for user {user_id}. Creating a new one...")
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = self.service.files().create(body=file_metadata, fields='id').execute()
            folder_id = folder.get('id')
            logger.info(f"Created new folder for user {user_id} with ID: {folder_id}")
            return folder_id
            
        except Exception as e:
            logger.error(f"Failed to get/create user folder: {e}", exc_info=True)
            raise

    def merge_and_encrypt_upload(self, chunk_paths, session_id, user_id):
        """Merges, encrypts, and uploads audio to Drive. (Blocking)"""
        self._check_service()
        try:
            logger.debug(f"Merging {len(chunk_paths)} audio chunks for session {session_id}...")
            merged_audio = b''
            for path in chunk_paths:
                with open(path, 'rb') as f:
                    merged_audio += f.read()
            
            logger.debug(f"Merged audio size: {len(merged_audio)} bytes.")
            logger.debug("Generating encryption key and encrypting merged audio...")
            key = Fernet.generate_key()
            fernet = Fernet(key)
            encrypted = fernet.encrypt(merged_audio)
            logger.debug("Encryption complete.")
            
            folder_id = self.get_user_folder(user_id)
            if not folder_id:
                raise Exception("Failed to get or create a user folder in Google Drive.")
            
            file_name = f'{session_id}.encrypted.wav'
            file_metadata = {
                'name': file_name,
                'parents': [folder_id]
            }
            media = MediaIoBaseUpload(io.BytesIO(encrypted), mimetype='audio/wav', resumable=True)
            
            logger.info(f"Uploading '{file_name}' to Google Drive folder ID: {folder_id}...")
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            file_id = file.get('id')
            logger.info(f"Successfully uploaded encrypted session {session_id} to Drive. File ID: {file_id}")
            
            return key.decode(), file_id
            
        except Exception as e:
            logger.error(f"Drive upload failed for session {session_id}: {e}", exc_info=True)
            return None, None

    def download_and_decrypt_audio(self, file_id, encryption_key):
        """Downloads and decrypts audio from Drive. (Blocking)"""
        self._check_service()
        try:
            logger.info(f"Starting download for Google Drive file ID: {file_id}")
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    logger.debug(f"Download progress: {int(status.progress() * 100)}%")
            
            logger.info("Download complete. Decrypting file...")
            fh.seek(0)
            encrypted_data = fh.read()
            fernet = Fernet(encryption_key.encode())
            decrypted_data = fernet.decrypt(encrypted_data)
            logger.info(f"File {file_id} decrypted successfully. Size: {len(decrypted_data)} bytes.")
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Failed to download and decrypt file {file_id}: {e}", exc_info=True)
            return None

    def find_file_in_folder(self, folder_name: str, file_name: str) -> str:
        """Finds a file by name within a specific folder. (Blocking)"""
        self._check_service()
        try:
            folder_query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            folder_response = self.service.files().list(
                q=folder_query,
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            
            folders = folder_response.get('files', [])
            if not folders:
                logger.warning(f"Could not find the parent folder '{folder_name}' on Google Drive.")
                return None
            
            folder_id = folders[0]['id']
            file_query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
            file_response = self.service.files().list(
                q=file_query,
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            
            files = file_response.get('files', [])
            if files:
                file_id = files[0]['id']
                logger.info(f"Found file '{file_name}' with ID '{file_id}' inside folder '{folder_name}'.")
                return file_id
            else:
                logger.warning(f"Could not find the file '{file_name}' inside the folder '{folder_name}'.")
                return None
                
        except Exception as e:
            logger.error(
                f"An error occurred while searching for file '{file_name}' in folder '{folder_name}': {e}",
                exc_info=True
            )
            return None