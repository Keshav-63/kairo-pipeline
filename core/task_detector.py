# core/task_detector.py
from config.logger import logger
from utils.langchain_llm_handler import LangchainLLMHandler
import json
from datetime import datetime

class TaskDetector:
    def __init__(self, llm_handler: LangchainLLMHandler):
        self.llm_handler = llm_handler

    async def detect_task(self, user_id: str, text_segment: str) -> dict:
        """
        Uses an LLM to detect if a text segment contains a task or reminder.
        Returns a structured dictionary if a task is found, otherwise None.
        """
        if not text_segment or not text_segment.strip():
            return None

        # The current time is needed to give the LLM context for relative dates like "tomorrow".
        current_time_iso = datetime.utcnow().isoformat()

        prompt = f"""
        Analyze the following text from a conversation to determine if it contains an actionable task, reminder, or deadline.
        The current time is {current_time_iso} UTC.

        Conversation Text:
        ---
        "{text_segment}"
        ---

        If a clear task, reminder, or deadline is present, respond with a single JSON object containing the following keys:
        - "is_task": (boolean) Must be true.
        - "task_description": (string) A concise description of the action item (e.g., "Email John about the report").
        - "due_date_time": (string) The specific due date and time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS). If the date is relative (e.g., "tomorrow", "next Friday at 5pm"), calculate the absolute ISO 8601 timestamp based on the current time provided above. If no specific time is mentioned, default to a reasonable time like 09:00 local time. If no date or time is mentioned at all, set this to null.
        - "assignee": (string) The person the task is assigned to. Default to "self" if not specified.
        - "context": (string) The original, full sentence where the task was mentioned.

        If NO actionable task, reminder, or deadline is found, respond with ONLY the following JSON object:
        {{
            "is_task": false
        }}

        Example Responses:
        - For "remind me to email John about the report tomorrow morning", your response should be something like:
          {{ "is_task": true, "task_description": "Email John about the report", "due_date_time": "2025-10-08T09:00:00", "assignee": "self", "context": "remind me to email John about the report tomorrow morning" }}
        - For "I need to pick up the dry cleaning on Friday.", your response should be something like:
          {{ "is_task": true, "task_description": "Pick up the dry cleaning", "due_date_time": "2025-10-10T09:00:00", "assignee": "self", "context": "I need to pick up the dry cleaning on Friday." }}
        - For "The weather is nice today.", your response must be:
          {{ "is_task": false }}

        Respond with ONLY the JSON object based on your analysis of the conversation text. Do not include any introductory or explanatory text outside the JSON structure.
        """

        try:
            task_json = await self.llm_handler.get_structured_llm_response(user_id, prompt)

            if task_json and task_json.get("is_task") is True:
                # Basic validation to ensure required fields are present
                if "task_description" in task_json and "due_date_time" in task_json:
                    logger.info(f"Task detected: {task_json.get('task_description')}")
                    return task_json
            return None
        except Exception as e:
            logger.error(f"Failed to detect task from LLM: {e}", exc_info=True)
            return None