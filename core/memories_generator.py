# kairo_pipeline/core/memories_generator.py

from config.logger import logger
from utils.langchain_llm_handler import LangchainLLMHandler # Assuming this is your LLM handler
import asyncio

class MemoriesGenerator:
    def __init__(self, llm_handler: LangchainLLMHandler):
        self.llm_handler = llm_handler

    async def generate_memory(self, user_id: str, full_transcription_text: str) -> dict:
        """
        Generates a structured "memory" from a full session transcript using an LLM.
        """
        if not full_transcription_text or not full_transcription_text.strip():
            logger.warning("Transcription text is empty. Cannot generate memory.")
            return None

        try:
            # This is a comprehensive prompt that asks the LLM to return a structured JSON object.
            # This is more reliable than asking for separate pieces of information.
            prompt = f"""
            Based on the following conversation transcript, please generate a detailed "memory" in a structured JSON format.

            Transcript:
            ---
            {full_transcription_text}
            ---

            Please provide the output as a single JSON object with the following keys and data types:
            - "title": (string) A concise, descriptive title for the conversation (e.g., "Discussing Q3 Marketing Strategy").
            - "summary": (string) A detailed, paragraph-long summary of the entire conversation.
            - "key_points": (array of strings) A list of the most important topics, decisions, and takeaways.
            - "action_items": (array of strings) A list of clear, actionable tasks or to-dos mentioned. If none, return an empty array.
            - "sentiment": (object) An analysis of the conversation's sentiment with "label" (e.g., "Positive", "Neutral", "Negative") and "justification" (a brief explanation for the sentiment).
            - "key_quotes": (array of objects) A list of 2-3 most impactful quotes, each with "quote" (the text) and "speaker".

            Example of a desired JSON output format:
            {{
                "title": "Planning for the Annual Tech Conference",
                "summary": "The team met to discuss the logistics and presentation topics for the upcoming annual tech conference. John will handle the booth setup, while Jane will finalize the presentation slides. The main topic of the presentation will be the impact of AI on software development.",
                "key_points": [
                    "Finalize presentation slides by next Friday.",
                    "John is responsible for the conference booth logistics.",
                    "The presentation will focus on AI in software development."
                ],
                "action_items": [
                    "Jane to send the final draft of the presentation slides to the team for review.",
                    "John to confirm the booth dimensions with the event organizers."
                ],
                "sentiment": {{
                    "label": "Productive",
                    "justification": "The conversation was collaborative and focused on clear deliverables and deadlines."
                }},
                "key_quotes": [
                    {{
                        "quote": "Let's make sure we have a compelling demo for the booth.",
                        "speaker": "Jane"
                    }},
                    {{
                        "quote": "I'll get the final budget numbers by end of day.",
                        "speaker": "John"
                    }}
                ]
            }}

            Now, generate ONLY the JSON object for the provided transcript. Do not include any introductory or explanatory text outside the JSON structure.
            """

            # This assumes your llm_handler has a method to get a structured JSON response.
            # You might need to adjust this call based on your LangchainLLMHandler implementation.
            memory_json = await self.llm_handler.get_structured_llm_response(user_id, prompt)

            if not memory_json:
                logger.error("LLM did not return a valid memory JSON object.")
                return None

            # You might want to add validation here to ensure all keys are present
            return memory_json

        except Exception as e:
            logger.critical(f"Failed to generate memory from LLM: {e}", exc_info=True)
            return None