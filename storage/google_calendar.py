# storage/google_calendar.py

import asyncio
import logging
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.auth.exceptions import RefreshError
from datetime import datetime

# Use the same logger as the rest of your application
from config.logger import logger

class GoogleCalendarHandler:
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
        logger.debug(f"GoogleCalendarHandler instantiated for user {user_id}")

    async def initialize_service(self):
        """
        Asynchronously loads credentials, refreshes them if necessary,
        saves the new token, and builds the Google Calendar service.
        """
        try:
            self.creds = Credentials.from_authorized_user_info(self.user_credentials_dict)
        except Exception as e:
            logger.error(f"Failed to load credentials for user {self.user_id}: {e}")
            raise ValueError(f"Invalid credentials format: {e}")

        # Check if token is expired and needs refresh
        if self.creds.expired and self.creds.refresh_token:
            logger.info(f"Google Calendar token for user {self.user_id} has expired. Refreshing...")
            try:
                # Run the synchronous refresh in a thread to avoid blocking asyncio
                await asyncio.to_thread(self.creds.refresh, GoogleAuthRequest())
                
                # Save the new credentials back to MongoDB
                if self.mongo_handler and self.user_id:
                    new_creds_dict = {
                        'token': self.creds.token,
                        'refresh_token': self.creds.refresh_token,
                        'token_uri': self.creds.token_uri,
                        'client_id': self.creds.client_id,
                        'client_secret': self.creds.client_secret,
                        'scopes': self.creds.scopes,
                        'expiry': self.creds.expiry.isoformat() # Store as ISO string
                    }
                    await self.mongo_handler.update_user_credentials(self.user_id, new_creds_dict)
                    logger.info(f"Successfully refreshed and saved Calendar token for user {self.user_id}")
                else:
                    logger.warning(f"Token for {self.user_id} refreshed, but mongo_handler or user_id is missing. Cannot save.")
            
            except RefreshError as e:
                logger.error(f"Failed to refresh Calendar token for user {self.user_id}: {e}", exc_info=True)
                raise IOError(f"Token refresh failed: {e}. User may need to re-authenticate.")
            except Exception as e:
                logger.error(f"An unexpected error occurred during Calendar token refresh for {self.user_id}: {e}", exc_info=True)
                raise
        
        elif not self.creds.valid:
             logger.warning(f"Calendar credentials for {self.user_id} are not valid and no refresh token is available.")
             raise ValueError("Invalid credentials and no refresh token.")

        # Build the service
        try:
            self.service = build('calendar', 'v3', credentials=self.creds)
            logger.info(f"Google Calendar service initialized for user {self.user_id}")
        except Exception as e:
            logger.error(f"Failed to build Google Calendar service: {e}", exc_info=True)
            raise

    def _blocking_event_creation(self, event_body):
        """
        The synchronous (blocking) Google API call to create an event.
        This method is intended to be run in a separate thread.
        """
        if not self.service:
            logger.error(f"Cannot create event: service for user {self.user_id} was not initialized.")
            return None
        try:
            event = self.service.events().insert(
                calendarId='primary',  # Use the user's primary calendar
                body=event_body,
                sendUpdates='all' # Send notifications to attendees (the user)
            ).execute()
            
            logger.info(f"Event created for user {self.user_id}: {event.get('htmlLink')}")
            return event.get('id')
            
        except HttpError as error:
            if error.resp.status == 403:
                 logger.error(f"Failed to create calendar event for {self.user_id}. "
                             f"Reason: 403 Forbidden. "
                             f"**CRITICAL**: Does the user have the 'calendar.events' scope? {error}",
                             exc_info=True)
            else:
                logger.error(f'An HttpError occurred creating calendar event for {self.user_id}: {error}', exc_info=True)
            return None
        except Exception as e:
            logger.error(f'An unexpected error occurred during event creation for {self.user_id}: {e}', exc_info=True)
            return None

    async def create_event(self, event_body):
        """
        Asynchronously creates a new event in the user's primary calendar.
        """
        if not self.service:
            logger.error(f"Cannot create event for {self.user_id}: Service not initialized. Did you call await initialize_service()?")
            return None
            
        try:
            # Run the blocking API call in a separate thread
            event_id = await asyncio.to_thread(self._blocking_event_creation, event_body)
            return event_id
            
        except Exception as e:
            logger.error(f"Generic error in create_event for {self.user_id}: {e}", exc_info=True)
            return None