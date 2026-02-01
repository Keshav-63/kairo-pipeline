from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient
from azure.storage.blob import BlobServiceClient
from config.logger import logger
import json

class AzureBlobHandler:
    def __init__(self, account_name, account_key, container):
        self.container_name = container
        self.account_url = f"https://{account_name}.blob.core.windows.net"
        self.account_key = account_key
        
        # Async client for the main pipeline
        self.async_client = AsyncBlobServiceClient(account_url=self.account_url, credential=self.account_key)
        self.async_container_client = self.async_client.get_container_client(container)
        
        # Sync client for the Flask API
        self.sync_client = BlobServiceClient(account_url=self.account_url, credential=self.account_key)
        self.sync_container_client = self.sync_client.get_container_client(container)

    def upload_voice_sample_sync(self, user_id: str, person_name: str, audio_data: bytes) -> str:
        """
        Synchronously uploads a voice sample for enrollment using the sync client.
        """
        try:
            blob_name = f"enrollments/{user_id}/{person_name}_sample.wav"
            blob_client = self.sync_container_client.get_blob_client(blob_name)
            
            blob_client.upload_blob(audio_data, overwrite=True)
            
            logger.info(f"Synchronously uploaded voice sample to {blob_name}")
            return blob_client.url
        except Exception as e:
            logger.error(f"Sync voice sample upload failed: {e}", exc_info=True)
            return None

    def download_voice_sample_sync(self, user_id: str, person_name: str) -> bytes:
        """
        Synchronously downloads a voice sample for a specific person.
        """
        try:
            blob_name = f"enrollments/{user_id}/{person_name}_sample.wav"
            blob_client = self.sync_container_client.get_blob_client(blob_name)
            data = blob_client.download_blob().readall()
            return data
        except Exception as e:
            logger.error(f"Failed to synchronously download voice sample for {person_name}: {e}", exc_info=True)
            return None

    async def list_user_sessions(self, user_id: str) -> list:
        """
        Asynchronously lists all session folders for a given user.
        """
        try:
            prefix = f"{user_id}_"
            logger.debug(f"Listing blobs with prefix: {prefix}")
            
            sessions = set()
            async for blob in self.async_container_client.list_blobs(name_starts_with=prefix):
                parts = blob.name.split('/')
                if len(parts) > 0:
                    sessions.add(parts[0])
            
            session_list = list(sessions)
            logger.debug(f"Found sessions for user {user_id}: {session_list}")
            return session_list
            
        except Exception as e:
            logger.error(f"Error listing Azure sessions for {user_id}: {e}", exc_info=True)
            return []

    async def download_session(self, session_id: str) -> (list, dict):
        """
        Asynchronously downloads all chunks and metadata for a given session.
        """
        try:
            prefix = f"{session_id}/"
            logger.info(f"Downloading files for session: {prefix}")
            
            blob_list_pager = self.async_container_client.list_blobs(name_starts_with=prefix)
            
            chunks = []
            metadata = None
            
            async for blob in blob_list_pager:
                blob_client = self.async_container_client.get_blob_client(blob.name)
                
                stream = await blob_client.download_blob()
                data = await stream.readall()
                
                if blob.name.endswith('.wav'):
                    chunks.append(data)
                elif blob.name.endswith('metadata.json'):
                    metadata = json.loads(data)
            
            logger.info(f"Downloaded {len(chunks)} audio chunks and metadata for session {session_id}")
            return chunks, metadata

        except Exception as e:
            logger.error(f"Failed to download session {session_id}: {e}", exc_info=True)
            return [], None
            
    async def download_voice_sample(self, user_id: str, person_name: str) -> bytes:
        """
        Asynchronously downloads a voice sample for a specific person.
        """
        try:
            blob_name = f"enrollments/{user_id}/{person_name}_sample.wav"
            blob_client = self.async_container_client.get_blob_client(blob_name)
            stream = await blob_client.download_blob()
            data = await stream.readall()
            return data
        except Exception as e:
            logger.error(f"Failed to asynchronously download voice sample for {person_name}: {e}", exc_info=True)
            return None

