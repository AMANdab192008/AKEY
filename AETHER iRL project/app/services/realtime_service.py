"""
REALTIME GROQ SERVICE MODULE
============================

Extends GroqService to add Tavily web search before calling the LLM. Used by
ChatService for POST /chat/realtime. Same session and history as general chat;
the only difference is we run a Tavily search for the user's question and add
the results to the system message, then call Groq.

ROUND-ROBIN API KEYS:
	- Shares the same round-robin counter as GroqService (class-level _shared_key_index)
	- This means /chat and /chat/realtime requests use the same rotation sequence
	- Example: If /chat uses key 1, the next /chat/realtime request will use key 2
	- All API key usage is logged with masked keys for security and debugging

FLOW:
	1. search_tavily(question): call Tavily API, format results as text (or "" on failure).
	2. get_response(question, chat_history): add search results to system message,
		 then same as parent: retrieve context from vector store, build prompt, call Groq.

If TAVILY_API_KEY is not set, tavily_client is None and search_tavily returns "";
the user still gets an answer from Groq with no search results.
"""

from typing import List, Optional
from tavily import TavilyClient
import logging
import os

from app.services.groq_service import GroqService, escape_curly_braces
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information
from app.utils.retry import with_retry
from config import AKEY_SYSTEM_PROMPT
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


logger = logging.getLogger("A.K.E.Y")


# ==================================================================================
# REALTIME GROQ SERVICE CLASS (extends GroqService)
# ==================================================================================

class RealtimeGroqService(GroqService):
	"""
	Same as GroqService but runs a Tavily web search first and adds the results
	to the system message. If Tavily is missing or fails, we still call Groq with
	no search results (user gets an answer without real-time data).
	"""

	def __init__(self, vector_store_service: VectorStoreService):
		"""Call parent init (Groq LLM + vector store); then create Tavily client if key is set."""
		super().__init__(vector_store_service)
		tavily_api_key = os.getenv("TAVILY_API_KEY", "")
		if tavily_api_key:
			self.tavily_client = TavilyClient(api_key=tavily_api_key)
			logger.info("Tavily search client initialized successfully")