from groq import Groq
from config.logger import logger

class LLMSynthesis:
    def __init__(self, api_key, model='llama-3.3-70b-versatile'):
        self.client = Groq(api_key=api_key)
        self.model = model
        # Test API key on init
        try:
            self.client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                model=self.model,
                max_tokens=1
            )
            logger.info("Groq API initialized successfully")
        except Exception as e:
            logger.error(f"Groq API initialization failed: {e}")
            raise

    def synthesize(self, query, contexts):
        try:
            if not contexts or not any(contexts):
                logger.warning("No valid contexts provided for synthesis.")
                return "No context available to generate a response."

            prompt = f"Query: {query}\nContexts: {' '.join(contexts)}\nSynthesize a concise, accurate answer:"
            logger.debug(f"Sending prompt to Groq: {prompt[:200]}...")  # Log prompt for debugging
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.5,
                max_tokens=500
            )
            
            logger.debug(f"Groq response: {chat_completion}")  # Log full response for debugging
            if not chat_completion.choices or not chat_completion.choices[0].message:
                logger.warning("Groq API returned empty or invalid choices.")
                return "Sorry, I couldn't generate a response at this time."

            answer = chat_completion.choices[0].message.content.strip() if chat_completion.choices[0].message.content else None
            logger.debug(f"Synthesized answer debugging...............: {answer}")  # Log the synthesized answer for debugging
            if not answer:
                logger.warning("Groq API returned empty content.")
                return "No meaningful response generated."

            logger.info(f"Synthesized answer for query: {query[:200]}... (Answer: {answer[:50]}...)")
            return answer
        except Exception as e:
            logger.error(f"Failed to synthesize answer: {e}", exc_info=True)
            return "Sorry, an error occurred while generating the response."