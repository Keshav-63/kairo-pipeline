# # kairo_pipeline/utils/langchain_llm_handler.py
# import os
# from config.logger import logger
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate 
# from langchain_groq import ChatGroq
# from langchain.chains import LLMChain # ADDED
# from langchain_core.messages import HumanMessage, AIMessage
# import yaml # To load Groq API key from settings.yaml
# import datetime
# from utils.time_parser import parse_time_query
# import json

# class LangchainLLMHandler:
#     def __init__(self, config_path: str):
#         self.config_path = config_path
#         self.llm = None
#         # self.user_chat_history = {} # Store chat history per user_id
#         self._initialize_llm()
#         self._setup_prompts()

#     def _initialize_llm(self):
#         """Initializes the ChatGroq LLM from configuration."""
#         try:
#             with open(self.config_path, 'r') as f:
#                 config = yaml.safe_load(f)
            
#             groq_api_key = config['groq']['api_key']
#             if not groq_api_key:
#                 raise ValueError("GROQ_API_KEY not found in config.")
            
#             self.llm = ChatGroq(
#                 temperature=0.3, 
#                 groq_api_key=groq_api_key, 
#                 model_name=config['groq'].get('model', 'llama-3.1-8b-instant')
#             )
#             logger.info(f"Langchain ChatGroq LLM initialized with model: {self.llm.model_name}")
#         except FileNotFoundError:
#             logger.critical(f"Configuration file not found at {self.config_path}. Cannot initialize LLM.")
#             raise
#         except KeyError:
#             logger.critical("Groq API key or model not found in configuration.")
#             raise
#         except Exception as e:
#             logger.critical(f"Failed to initialize Langchain ChatGroq LLM: {e}", exc_info=True)
#             raise

#     def _setup_prompts(self):
#         """Sets up the Langchain prompt templates."""
#         # History-aware prompt
#         self.contextualize_q_system_prompt = """
#         You are KAIRO, the AI smart pendant’s assistant.

#         Rules:
#         - Reformulate the user’s question into a **standalone question** without extra text.
#         - Output only the final question in plain text — no explanations or preamble.
#         - Ensure clarity and context independence so the question can be answered without prior chat history.
#         - Keep it concise and precise.

#         Example:
#         User input: "facebook yesterday"
#         Output: "What discussions happened on Facebook yesterday?"
#         """


#         self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", self.contextualize_q_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )

#         # Main answer generation prompt
#         # In langchain_llm_handler.py, update the qa_system_prompt in _setup_prompts():
#         self.qa_system_prompt = """
# You are KAIRO, an AI memory assistant in a smart pendant. Help recall conversations with details based *only* on the provided context.

# Context:
# ---
# {context}
# ---

# Rules:
# - Base your answer **strictly** on the provided Context section above. Do not use prior knowledge.

# - **CRITICAL RULE:** The user's query (e.g., "what happened yesterday?" or "what did we say about {{topic}} on {{date}}?") was used to find the {context} above.
# - You MUST assume the {context} provided IS the correct and relevant information for the user's query.
# - **DO NOT** state that the query's terms (like a specific date, time, or topic) are "not mentioned" in the context. Your task is to **summarize the context** that was found.
# - **For example:**
#     - If the user asks "What did we discuss on October 3rd?" and the {context} contains "...talked about UPI and Aadhaar...", your answer MUST be: "On October 3rd, you discussed UPI and Aadhaar...".
#     - You must **connect** the query to the context, even if the context doesn't explicitly repeat the query's terms.

# - **Fallback:** If, and *only if*, the Context section is empty or contains only a message like "No recorded contexts found...", then explain that no information was found (e.g., "I couldn't find recordings matching your request.").

# - Respond in **Markdown** with sections: **Summary**, **Key Points** (bullet list), **Quotes** (blockquotes if available), **Sources** (mention the sessions or relevant parts of the context), **Next Steps** (e.g., suggest follow-ups based on the context).
# - Highlight relevant parts from the context using bold text.

# Chat History: {chat_history}
# User Question: {input}
#     """
#         # self.qa_system_prompt = """
#         # You are KAIRO, an AI assistant embedded in a smart pendant device designed to recall conversations and provide rich, detailed answers.

#         # Rules:
#         # - Provide the answer in **Markdown format**.
#         # - Use headings, bullet points, numbered lists, quotes, bold, italics, and code blocks where needed.
#         # - Include point-wise explanations where applicable.
#         # - Highlight important keywords with bold.
#         # - If relevant, include direct quotes from the source inside blockquotes (`>`).
#         # - Do NOT include apologies or unnecessary preamble.
#         # - Keep the answer concise but detailed enough so it feels like an in-depth answer from your own reasoning.
#         # - If there is insufficient information, respond: "I don’t have enough information on this topic."

#         # Example Markdown Output:
#         # Answer Summary

#         # Key Point 1: Explanation of point 1.

#         # Key Point 2: Explanation of point 2.

#         # "Direct quote from context if available."

#         # Conclusion: Final concise thought.

#         # Context: {context}
#         # Chat History: {chat_history}
#         # User Question: {input}

#         # Please generate your answer in this Markdown style strictly.
#         # """

#         self.qa_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", self.qa_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )
#         logger.info("Langchain prompt templates set up.")
        
        
        
#     def _format_chat_history(self, chat_history_from_db: list) -> list:
#         """
#         Formats the chat history from the database (list of dicts)
#         into a list of Langchain Message objects.
#         """
#         if not chat_history_from_db:
#             return []
        
#         formatted_history = []
#         for message in chat_history_from_db:
#             role = message.get('role')
#             content = message.get('content')
#             if role == 'user':
#                 formatted_history.append(HumanMessage(content=content))
#             elif role == 'assistant':
#                 formatted_history.append(AIMessage(content=content))
#         return formatted_history



#   # --- MODIFIED METHOD ---
#     async def get_standalone_question(self, user_id: str, query_text: str, chat_history: list = None) -> str:
#         """
#         Generates a standalone question from the user's query and the provided chat history.
#         """
#         try:
#             # Format the history from DB to Langchain messages
#             formatted_chat_history = self._format_chat_history(chat_history)
            
#             history_aware_chain = self.contextualize_q_prompt | self.llm
#             standalone_question_response = await history_aware_chain.ainvoke({
#                 "chat_history": formatted_chat_history,
#                 "input": query_text
#             })
#             standalone_question = standalone_question_response.content
#             logger.debug(f"Standalone question for user {user_id}: {standalone_question}")
#             return standalone_question
#         except Exception as e:
#             logger.error(f"Failed to generate standalone question for user {user_id}: {e}", exc_info=True)
#             return query_text # Fallback to original query on error

#     # --- MODIFIED METHOD ---
#     async def get_answer_from_context(self, user_id: str, query_text: str, context_text: str, chat_history: list = None) -> str:
#         """
#         Generates an answer using the provided context, user query, and chat history.
#         """
#         try:
#             # Format the history from DB to Langchain messages
#             formatted_chat_history = self._format_chat_history(chat_history)
#             # Get time from parse_time_query result (assuming it's stored or accessible)
#             start_ts, end_ts = parse_time_query(query_text)  # Re-use or fetch from earlier
#             time_value = datetime.datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d') if start_ts else "No specific time"
            
#             qa_chain = self.qa_prompt | self.llm
#             logger.debug(f"Invoking QA chain for user {user_id} with context length {len(context_text)}...")
#             logger.debug(f"FULL CONTEXT TEXT FOR DEBUGGING: {self.qa_prompt}...") 
#             answer_response = await qa_chain.ainvoke({
#                 "context": context_text,
#                 "chat_history": formatted_chat_history,
#                 "input": query_text,
#                 "time": time_value
#             })
#             answer = answer_response.content
#             logger.debug(f"Generated answer for user {user_id}: {answer[:100]}...")
#             return answer
#         except Exception as e:
#             logger.error(f"Failed to generate answer from context for user {user_id}: {e}", exc_info=True)
#             return "Sorry, an error occurred while generating the response from the LLM."
        
        
# # kairo_pipeline/utils/langchain_llm_handler.py

#     async def generate_chat_title(self, chat_history: list) -> str:
#         """
#         Generates a concise title for a chat session based on its history.
#         """
#         if not chat_history:
#             return "New Chat"

#         # Format the history for the prompt
#         formatted_history = "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in chat_history])
        
#         # Define the prompt template string directly
#         prompt_template = (
#             "Based on the following conversation, create a very short, concise title (4-5 words maximum) that summarizes the main topic.\n\n"
#             "Conversation:\n{chat_history}\n\n"
#             "Title:"
#         )
        
#         # Create the PromptTemplate and LLMChain
#         prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history"])
#         chain = LLMChain(llm=self.llm, prompt=prompt)
        
#         try:
#             # Use chain.arun for async execution
#             response = await chain.arun({"chat_history": formatted_history})
            
#             # Clean up the response, removing quotes and extra whitespace
#             title = response.strip().replace('"', '')
#             logger.info(f"Generated chat title: '{title}'")
#             return title if title else "Chat Summary"
#         except Exception as e:
#             logger.error(f"Failed to generate chat title with LLM: {e}", exc_info=True)
#             return "Chat Summary" # Fallback title




#         # --- START: ADDED MISSING METHODS ---

#     async def get_answer(self, user_id: str, prompt_text: str) -> str:
#         """
#         A generic method to get a direct answer from the LLM for any given prompt text.
#         """
#         try:
#             logger.debug(f"Invoking generic LLM for user {user_id} with prompt: {prompt_text[:150]}...")
#             # Create a simple, general-purpose chain for this call
#             prompt = PromptTemplate(template="{prompt}", input_variables=["prompt"])
#             chain = LLMChain(llm=self.llm, prompt=prompt)
#             response = await chain.arun({"prompt": prompt_text})
#             return response
#         except Exception as e:
#             logger.error(f"Error in generic get_answer for user {user_id}: {e}", exc_info=True)
#             return "" # Return empty string on error

#     async def get_structured_llm_response(self, user_id: str, prompt: str) -> dict:
#         """
#         Gets a response from the LLM and attempts to parse it as a JSON object.
#         """
#         try:
#             # Use the new get_answer method
#             full_response = await self.get_answer(user_id, prompt)

#             if not full_response:
#                 return None

#             # The LLM might return the JSON wrapped in markdown ```json ... ```, so we clean it.
#             if "```json" in full_response:
#                 json_str = full_response.split("```json")[1].split("```")[0].strip()
#             else:
#                 json_str = full_response

#             return json.loads(json_str)
#         except json.JSONDecodeError as e:
#             logger.error(f"Failed to decode LLM response into JSON: {e}. Response was: {full_response}")
#             return None
#         except Exception as e:
#             logger.error(f"An error occurred while getting structured LLM response: {e}", exc_info=True)
#             return None

#     # --- END: ADDED MISSING METHODS ---


#     # def update_chat_history(self, user_id: str, human_message: str, ai_message: str):
#     #     """Updates the chat history for a specific user."""
#     #     if user_id not in self.user_chat_history:
#     #         self.user_chat_history[user_id] = []
#     #     self.user_chat_history[user_id].append(HumanMessage(content=human_message))
#     #     self.user_chat_history[user_id].append(AIMessage(content=ai_message))
#     #     logger.debug(f"Chat history updated for user {user_id}. Current length: {len(self.user_chat_history[user_id])}")

#     # def clear_chat_history(self, user_id: str):
#     #     """Clears the chat history for a specific user."""
#     #     if user_id in self.user_chat_history:
#     #         del self.user_chat_history[user_id]
#     #         logger.info(f"Chat history cleared for user {user_id}.")





#-------------------------------------------------------------------------------------------
# # kairo_pipeline/utils/langchain_llm_handler.py
# import os
# from config.logger import logger
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate 
# from langchain_groq import ChatGroq
# from langchain.chains import LLMChain
# from langchain_core.messages import HumanMessage, AIMessage
# import yaml
# import datetime
# # --- FIX: Removed redundant time_parser import ---
# # from utils.time_parser import parse_time_query 
# import json

# class LangchainLLMHandler:
#     def __init__(self, config_path: str):
#         self.config_path = config_path
#         self.llm = None
#         self._initialize_llm()
#         self._setup_prompts()

#     def _initialize_llm(self):
#         """Initializes the ChatGroq LLM from configuration."""
#         try:
#             with open(self.config_path, 'r') as f:
#                 config = yaml.safe_load(f)
            
#             groq_api_key = config['groq']['api_key']
#             if not groq_api_key:
#                 raise ValueError("GROQ_API_KEY not found in config.")
            
#             self.llm = ChatGroq(
#                 temperature=0.3, 
#                 groq_api_key=groq_api_key, 
#                 model_name=config['groq'].get('model', 'llama-3.1-8b-instant')
#             )
#             logger.info(f"Langchain ChatGroq LLM initialized with model: {self.llm.model_name}")
#         except FileNotFoundError:
#             logger.critical(f"Configuration file not found at {self.config_path}. Cannot initialize LLM.")
#             raise
#         except KeyError:
#             logger.critical("Groq API key or model not found in configuration.")
#             raise
#         except Exception as e:
#             logger.critical(f"Failed to initialize Langchain ChatGroq LLM: {e}", exc_info=True)
#             raise

#     def _setup_prompts(self):
#         """Sets up the Langchain prompt templates."""
        
#         # --- FIX: Updated prompt to be stricter and less "creative" ---
#         self.contextualize_q_system_prompt = """
#         You are an AI assistant. Given a chat history and a follow-up question, your task is to rephrase the follow-up question into a standalone question.
        
#         Rules:
#         - Your *only* goal is to make the question understandable *without* the chat history.
#         - **Do NOT** "improve" or "expand" the user's question.
#         - **Do NOT** infer intent or add new topics (like "traditions and customs" if the user just asked "about diwali").
#         - If the question is already standalone, output it exactly as is.
#         - Output *only* the final question.

#         Example 1 (Needs context):
#         Chat History:
#         Human: We discussed project alpha yesterday.
#         AI: Yes, we did.
#         Follow-up Question: tell me more about that
#         Output: Tell me more about project alpha.

#         Example 2 (Already standalone):
#         Chat History:
#         Human: what we are taking about diwali
#         AI: We discussed Diwali, a Hindu festival.
#         Follow-up Question: what task i suppose to tell you to remind on oct 7 2025...
#         Output: what task i suppose to tell you to remind on oct 7 2025...
#         """
#         # --- END FIX ---


#         self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", self.contextualize_q_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )

#         # Main answer generation prompt
#         self.qa_system_prompt = """
# You are KAIRO, an AI memory assistant in a smart pendant. Help recall conversations with details based *only* on the provided context.

# Context:
# ---
# {context}
# ---

# Rules:
# - Base your answer **strictly** on the provided Context section above. Do not use prior knowledge.

# - **CRITICAL RULE:** The user's query (e.g., "what happened yesterday?" or "what did we say about {{topic}} on {{date}}?") was used to find the {context} above.
# - You MUST assume the {context} provided IS the correct and relevant information for the user's query.
# - **DO NOT** state that the query's terms (like a specific date, time, or topic) are "not mentioned" in the context. Your task is to **summarize the context** that was found.
# - **For example:**
#     - If the user asks "What did we discuss on October 7th?" and the {context} contains "keshav: remind me... project update... tomorrow.", your answer MUST be: "On October 7th, you mentioned a reminder to show a project update to Shreyansh tomorrow."
#     - You must **connect** the query to the context, even if the context doesn't explicitly repeat the query's terms.

# - **Fallback:** If, and *only if*, the Context section is empty or contains only a message like "No recorded contexts found...", then explain that no information was found (e.g., "I couldn't find recordings matching your request.").

# - Respond in **Markdown** with sections: **Summary**, **Key Points** (bullet list), **Quotes** (blockquotes if available), **Sources** (mention the sessions or relevant parts of the context), **Next Steps** (e.g., suggest follow-ups based on the context).
# - Highlight relevant parts from the context using bold text.

# Chat History: {chat_history}
# User Question: {input}
#     """
       
#         self.qa_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", self.qa_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )
#         logger.info("Langchain prompt templates set up.")
        
        
        
#     def _format_chat_history(self, chat_history_from_db: list) -> list:
#         """
#         Formats the chat history from the database (list of dicts)
#         into a list of Langchain Message objects.
#         """
#         if not chat_history_from_db:
#             return []
        
#         formatted_history = []
#         for message in chat_history_from_db:
#             role = message.get('role')
#             content = message.get('content')
#             if role == 'user':
#                 formatted_history.append(HumanMessage(content=content))
#             elif role == 'assistant':
#                 formatted_history.append(AIMessage(content=content))
#         return formatted_history



#     async def get_standalone_question(self, user_id: str, query_text: str, chat_history: list = None) -> str:
#         """
#         Generates a standalone question from the user's query and the provided chat history.
#         """
#         try:
#             formatted_chat_history = self._format_chat_history(chat_history)
            
#             history_aware_chain = self.contextualize_q_prompt | self.llm
#             standalone_question_response = await history_aware_chain.ainvoke({
#                 "chat_history": formatted_chat_history,
#                 "input": query_text
#             })
#             standalone_question = standalone_question_response.content
#             logger.debug(f"Standalone question for user {user_id}: {standalone_question}")
#             return standalone_question
#         except Exception as e:
#             logger.error(f"Failed to generate standalone question for user {user_id}: {e}", exc_info=True)
#             return query_text # Fallback to original query on error

#     async def get_answer_from_context(self, user_id: str, query_text: str, context_text: str, chat_history: list = None) -> str:
#         """
#         Generates an answer using the provided context, user query, and chat history.
#         """
#         try:
#             formatted_chat_history = self._format_chat_history(chat_history)
            
#             # --- FIX: Removed redundant call to parse_time_query ---
#             # --- and removed unused 'time_value' variable ---
            
#             qa_chain = self.qa_prompt | self.llm
#             logger.debug(f"Invoking QA chain for user {user_id} with context length {len(context_text)}...")
#             logger.debug(f"Invoking QA chain with query: {query_text}")
#             logger.debug(f"Context snippet: {context_text[:200]}...")

#             answer_response = await qa_chain.ainvoke({
#                 "context": context_text,
#                 "chat_history": formatted_chat_history,
#                 "input": query_text,
#                 # "time": time_value # <-- This variable is not in the prompt
#             })
#             answer = answer_response.content
#             logger.debug(f"Generated answer for user {user_id}: {answer[:100]}...")
#             return answer
#         except Exception as e:
#             logger.error(f"Failed to generate answer from context for user {user_id}: {e}", exc_info=True)
#             return "Sorry, an error occurred while generating the response from the LLM."
        
        
#     async def generate_chat_title(self, chat_history: list) -> str:
#         """
#         Generates a concise title for a chat session based on its history.
#         """
#         if not chat_history:
#             return "New Chat"

#         formatted_history = "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in chat_history])
        
#         prompt_template = (
#             "Based on the following conversation, create a very short, concise title (4-5 words maximum) that summarizes the main topic.\n\n"
#             "Conversation:\n{chat_history}\n\n"
#             "Title:"
#         )
        
#         prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history"])
#         chain = LLMChain(llm=self.llm, prompt=prompt)
        
#         try:
#             response = await chain.arun({"chat_history": formatted_history})
#             title = response.strip().replace('"', '')
#             logger.info(f"Generated chat title: '{title}'")
#             return title if title else "Chat Summary"
#         except Exception as e:
#             logger.error(f"Failed to generate chat title with LLM: {e}", exc_info=True)
#             return "Chat Summary" # Fallback title


#     async def get_answer(self, user_id: str, prompt_text: str) -> str:
#         """
# A generic method to get a direct answer from the LLM for any given prompt text.
# """
#         try:
#             logger.debug(f"Invoking generic LLM for user {user_id} with prompt: {prompt_text[:150]}...")
#             prompt = PromptTemplate(template="{prompt}", input_variables=["prompt"])
#             chain = LLMChain(llm=self.llm, prompt=prompt)
#             response = await chain.arun({"prompt": prompt_text})
#             return response
#         except Exception as e:
#             logger.error(f"Error in generic get_answer for user {user_id}: {e}", exc_info=True)
#             return "" # Return empty string on error

#     async def get_structured_llm_response(self, user_id: str, prompt: str) -> dict:
#         """
#         Gets a response from the LLM and attempts to parse it as a JSON object.
#         """
#         try:
#             full_response = await self.get_answer(user_id, prompt)

#             if not full_response:
#                 return None

#             if "```json" in full_response:
#                 json_str = full_response.split("```json")[1].split("```")[0].strip()
#             else:
#                 json_str = full_response

#             return json.loads(json_str)
#         except json.JSONDecodeError as e:
#             logger.error(f"Failed to decode LLM response into JSON: {e}. Response was: {full_response}")
#             return None
#         except Exception as e:
#             logger.error(f"An error occurred while getting structured LLM response: {e}", exc_info=True)
#             return None

#-----------------------------------------------
# kairo_pipeline/utils/langchain_llm_handler.py
import os
from config.logger import logger
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate 
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage
import yaml
import datetime
import json
import pytz # <--- IMPORT PYTZ

# --- ADD THIS ---
# Define the user's timezone, same as in time_parser.py
USER_TIMEZONE = pytz.timezone("Asia/Kolkata")

class LangchainLLMHandler:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.llm = None
        self._initialize_llm()
        self._setup_prompts()

    def _initialize_llm(self):
        """Initializes the ChatGroq LLM from configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            groq_api_key = config['groq']['api_key']
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not found in config.")
            
            self.llm = ChatGroq(
                temperature=0.6, 
                groq_api_key=groq_api_key, 
                model_name=config['groq'].get('model', 'llama-3.3-70b-versatile')
            )
            logger.info(f"Langchain ChatGroq LLM initialized with model: {self.llm.model_name}")
        except FileNotFoundError:
            logger.critical(f"Configuration file not found at {self.config_path}. Cannot initialize LLM.")
            raise
        except KeyError:
            logger.critical("Groq API key or model not found in configuration.")
            raise
        except Exception as e:
            logger.critical(f"Failed to initialize Langchain ChatGroq LLM: {e}", exc_info=True)
            raise

    def _setup_prompts(self):
        """Sets up the Langchain prompt templates."""
        
        self.contextualize_q_system_prompt ="""
You are KAIRO, an AI memory assistant embedded in a smart pendant. Your task is to take a follow-up question and produce a standalone question.

Rules:
- Keep the original wording of the user's question exactly as much as possible.
- Only add minimal context if pronouns or references are ambiguous.
- Do NOT infer new topics, or rephrase unnecessarily.
- Output only the final standalone question.

Examples:
- Chat: Human: We discussed project alpha yesterday. AI: Yes we did. Follow-up: tell me more about that
  Output: Tell me more about project alpha.
- Chat: Human: what we are taking about diwali. AI: We discussed Diwali. Follow-up: what task i suppose to tell you to remind on oct 7 2025...
  Output: what task i suppose to tell you to remind on oct 7 2025...
"""
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Main answer generation prompt
        self.qa_system_prompt =self.qa_system_prompt = """
You are **KAIRO**, an AI memory assistant embedded in a smart pendant. You have access to the user's conversation history and memory.

Your goal is to answer the user's query using the provided context, while making the answer:

- Rich in **Markdown** (bold, italics, bullet points, numbered lists, blockquotes)  
- Dynamic and varied in wording, so it doesn't repeat the same phrases  
- Clear, structured, and readable, with line breaks between sections  

# Guidelines:

**Sections to include (if applicable)**:

**Summary**  
- Provide a concise answer to the user's question.  
- Avoid repeating session IDs; just include relevant info like date, speaker, topic.

**Details / Quotes**  
- Include relevant conversation snippets.  
- Format as: **Speaker Name**: "Quote"  
- Maintain proper line spacing for readability.

**Inferences / Notes**  
- Clearly mark any inferred or likely information, e.g., "It seems that..."  
- Explain context or connections if user asked follow-up questions.

**Tasks / Reminders**  
- List any tasks, events, or reminders discussed, if applicable.

**Dates / Topics**  
- Include discussion dates or important topics mentioned in the conversation.

**Behavior Rules**:

1. Use **bold**, *italics*, bullet points, numbered lists, and blockquotes naturally.  
2. Vary the wording dynamically; avoid producing the same phrasing every time.  
3. If no relevant context is found, respond naturally: "I couldn't find relevant information in your recordings."  
4. Use the provided `{query_date}` if a date is explicitly mentioned in the query. Otherwise, infer from context.  
5. Answer the query fully, even if it’s a follow-up or partially ambiguous.  
6. Maintain **KAIRO’s personality**: helpful, accurate, memory-aware, conversational.  

# Context:
---
{context}
---

**User Query**: {input}  
**Query Date**: {query_date}

Respond in **Markdown**, following the structure above if applicable, with line breaks between sections.
"""

       
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        logger.info("Langchain prompt templates set up.")
        
        
        
    def _format_chat_history(self, chat_history_from_db: list) -> list:
        """
        Formats the chat history from the database (list of dicts)
        into a list of Langchain Message objects.
        """
        if not chat_history_from_db:
            return []
        
        formatted_history = []
        for message in chat_history_from_db:
            role = message.get('role')
            content = message.get('content')
            if role == 'user':
                formatted_history.append(HumanMessage(content=content))
            elif role == 'assistant':
                formatted_history.append(AIMessage(content=content))
        return formatted_history



    async def get_standalone_question(self, user_id: str, query_text: str, chat_history: list = None) -> str:
        """
        Generates a standalone question from the user's query and the provided chat history.
        """
        try:
            formatted_chat_history = self._format_chat_history(chat_history)
            
            history_aware_chain = self.contextualize_q_prompt | self.llm
            standalone_question_response = await history_aware_chain.ainvoke({
                "chat_history": formatted_chat_history,
                "input": query_text
            })
            standalone_question = standalone_question_response.content
            logger.debug(f"Standalone question for user {user_id}: {standalone_question}")
            return standalone_question
        except Exception as e:
            logger.error(f"Failed to generate standalone question for user {user_id}: {e}", exc_info=True)
            return query_text # Fallback to original query on error

    async def get_answer_from_context(
        self, 
        user_id: str, 
        query_text: str, 
        context_text: str, 
        chat_history: list = None, 
        start_ts: int = None
    ) -> str:
        """
        Generates an answer using the provided context, user query, and chat history.
        """
        try:
            formatted_chat_history = self._format_chat_history(chat_history)
            
            # --- THIS IS THE FIX IN langchain_llm_handler.get_answer_from_context() ---
            query_date_str = "No specific date"
            if start_ts:
                try:
                    # Treat start_ts as an epoch second (may be UTC or local depending on storage).
                    # We will avoid forcing UTC conversion; instead produce a localized date string using Asia/Kolkata.
                    dt = datetime.datetime.fromtimestamp(int(start_ts))
                    # If dt is naive, localize to USER_TIMEZONE (so it becomes Asia/Kolkata)
                    if dt.tzinfo is None:
                        dt = USER_TIMEZONE.localize(dt)
                    else:
                        dt = dt.astimezone(USER_TIMEZONE)
                    query_date_str = dt.strftime('%B %d, %Y')
                except Exception as e:
                    logger.warning(f"Could not parse timestamp {start_ts}: {e}")
            # --- END FIX ---

            
            qa_chain = self.qa_prompt | self.llm
            logger.debug(f"Invoking QA chain for user {user_id} with context length {len(context_text)}...")
            logger.debug(f"Invoking QA chain with query: {query_text} and date: {query_date_str}") # This will now be correct
            logger.debug(f"Context snippet: {context_text[:200]}...")

            answer_response = await qa_chain.ainvoke({
                "context": context_text,
                "chat_history": formatted_chat_history,
                "input": query_text,
                "query_date": query_date_str
            })
            answer = answer_response.content
            logger.debug(f"Generated answer for user {user_id}: {answer[:100]}...")
            return answer
        except Exception as e:
            logger.error(f"Failed to generate answer from context for user {user_id}: {e}", exc_info=True)
            return "Sorry, an error occurred while generating the response from the LLM."
        
        
    async def generate_chat_title(self, chat_history: list) -> str:
        """
        Generates a concise title for a chat session based on its history.
        """
        if not chat_history:
            return "New Chat"

        formatted_history = "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in chat_history])
        
        prompt_template = (
            "Based on the following conversation, create a very short, concise title (4-5 words maximum) that summarizes the main topic.\n\n"
            "Conversation:\n{chat_history}\n\n"
            "Title:"
        )
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            response = await chain.arun({"chat_history": formatted_history})
            title = response.strip().replace('"', '')
            logger.info(f"Generated chat title: '{title}'")
            return title if title else "Chat Summary"
        except Exception as e:
            logger.error(f"Failed to generate chat title with LLM: {e}", exc_info=True)
            return "Chat Summary" # Fallback title


    async def get_answer(self, user_id: str, prompt_text: str) -> str:
        """
        A generic method to get a direct answer from the LLM for any given prompt text.
        """
        try:
            logger.debug(f"Invoking generic LLM for user {user_id} with prompt: {prompt_text[:150]}...")
            prompt = PromptTemplate(template="{prompt}", input_variables=["prompt"])
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = await chain.arun({"prompt": prompt_text})
            return response
        except Exception as e:
            logger.error(f"Error in generic get_answer for user {user_id}: {e}", exc_info=True)
            return "" # Return empty string on error

    # async def get_structured_llm_response(self, user_id: str, prompt: str) -> dict:
    #     """
    #     Gets a response from the LLM and attempts to parse it as a JSON object.
    #     """
    #     try:
    #         full_response = await self.get_answer(user_id, prompt)

    #         if not full_response:
    #             return None

    #         if "```json" in full_response:
    #             json_str = full_response.split("```json")[1].split("```")[0].strip()
    #         else:
    #             json_str = full_response

    #         return json.loads(json_str)
    #     except json.JSONDecodeError as e:
    #         logger.error(f"Failed to decode LLM response into JSON: {e}. Response was: {full_response}")
    #         return None
    #     except Exception as e:
    #         logger.error(f"An error occurred while getting structured LLM response: {e}", exc_info=True)
    #         return None
    async def get_structured_llm_response(self, user_id: str, prompt: str) -> dict:
        """
        Gets a response from the LLM and robustly attempts to parse it as a JSON object,
        even if wrapped in markdown or extra text.
        """
        try:
            full_response = await self.get_answer(user_id, prompt)

            if not full_response:
                logger.warning("LLM returned an empty response for structured request.")
                return None

            # Attempt to find the JSON block, handling potential markdown wrappers
            json_str = None
            if "```json" in full_response:
                # Try extracting from ```json block first
                 try:
                    json_str = full_response.split("```json")[1].split("```")[0].strip()
                 except IndexError:
                    logger.warning("Found ```json marker but failed to extract content.")
                    json_str = None # Fallback to next method

            if json_str is None:
                 # If no ```json or extraction failed, find the first '{' and last '}'
                 first_brace = full_response.find('{')
                 last_brace = full_response.rfind('}')
                 if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                     json_str = full_response[first_brace:last_brace+1].strip()
                 else:
                     # If braces aren't found, maybe the whole response is JSON? Try it.
                     json_str = full_response.strip()

            if not json_str:
                logger.error(f"Could not extract a potential JSON string from LLM response: {full_response}")
                return None

            # Attempt to parse the extracted string
            return json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode LLM response into JSON: {e}. Extracted string was: '{json_str}'. Full response: {full_response}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while getting structured LLM response: {e}", exc_info=True)
            return None