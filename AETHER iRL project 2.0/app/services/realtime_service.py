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

from typing import List, Optional, Iterator, Tuple, Any
from tavily import TavilyClient
import logging
import os
import time

from app.services.groq_service import GroqService, escape_curly_braces, AllGroqApisFailedError
from app.services.vector_store import VectorStoreService
from app.utils.retry import with_retry
from config import REALTIME_CHAT_ADDENDUM, GROQ_API_KEYS, GROQ_MODEL


logger = logging.getLogger("A.K.E.Y")

GROQ_REQUEST_TIMEOUT_FAST = 15
_QUERY_EXTRACTION_PROMPT = (
    "You are a search query optimizer. Given the user's message and recent conversation, "
    "produce a single short, focused web search query (max 12 words) that will find the "
    "information the user needs. Resolve any references (like 'that website', 'him', 'it') "
    "using the conversation history. Output ONLY the search query, nothing else."
)

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
        else:
            self.tavily_client = None
            logger.warning("TAVILY_API_KEY not set. Realtime search will be unavailable.")

        if GROQ_API_KEYS:
            from langchain_groq import ChatGroq
            self._fast_llm = ChatGroq(
                groq_api_key=GROQ_API_KEYS[0],
                model_name=GROQ_MODEL,
                temperature=0.0,
                request_timeout=GROQ_REQUEST_TIMEOUT_FAST,
                max_tokens=50,
            )
        else:
            self._fast_llm = None

    def _extract_search_query(
        self, question: str, chat_history: Optional[List[tuple]] = None
    ) -> str:
        if not self._fast_llm:
            return question

        try:
            t0 = time.perf_counter()
            history_context = ""
            if chat_history:
                recent = chat_history[-3:]
                parts = []
                for h, a in recent:
                    parts.append(f"user: {h[:200]}")
                    parts.append(f"Assistant: {a[:200]}")
                history_context = "\n".join(parts)

            if history_context:
                full_prompt = (
                    f"{_QUERY_EXTRACTION_PROMPT}\n\n"
                    f"Recent conversation:\n{history_context}\n\n"
                    f"User's latest message: {question}\n\n"
                    f"Search query:"
                )
            else:
                full_prompt = (
                    f"{_QUERY_EXTRACTION_PROMPT}\n\n"
                    f"User's message: {question}\n\n"
                    f"Search query:"
                )

            response = self._fast_llm.invoke(full_prompt)
            extracted = response.content.strip().strip('"').strip("'")
            if extracted and 3 <= len(extracted) <= 200:
                logger.info(
                    "[REALTIME] Query extraction: '%s' -> '%s' (%.3fs)",
                    question[:80], extracted[:80],  time.perf_counter() - t0,
                )
                return extracted
            logger.warning("[REALTIME] Query extraction returned unusable result, using raw question")
            return question
        except Exception as e:
            logger.warning("[REALTIME] Query extraction failed (%s), using raw question" , e)
            return question
            
    def search_tavily(self, query: str, num_results: int = 7) -> str:
        """
        Call Tavily API with the given query and return formatted result text for the prompt.
        On any failure (no key, rate limit, network) we return "" so the LLM still gets a reply.
        """
        if not self.tavily_client:
            logger.warning("Tavily client not initialized. TAVILY_API_KEY not set.")
            return ("", None)
        
        try:
            t0 = time.perf_counter()
            # Perform Tavily search with retries for rate limits and transient errors.
            response = with_retry(
                lambda: self.tavily_client.search(
                    query=query,
                    search_depth="advanced", # "basic" is faster, "advanced" is more thorough
                    max_results=num_results,
                    include_answer=True, # We'll format our own results
                    include_raw_content=False, # Don't need full page content
                ),
                max_retries=3,
                initial_delay=1.0,
            )
            
            results = response.get('results', [])
            ai_answer = response.get("answer", "")
        
            if not results and not ai_answer:
                logger.warning("No Tavily search results found for query: %s", query)
                return ("", None)

            payload: Optional[dict] = {
                "query": query,
                "answer": ai_answer,
                "result": [
                    {
                        "title": r.get("title", "No title"),
                        "content": (r.get("content") or "")[:500],
                        "url": r.get("url", ""),
                        "score": round(float(r.get("score", 0)), 2),
                    }
                    for r in results[:num_results]
                ],
            }

            # Format results into a structured text block for the LLM.
            parts = [f"=== WEB SEARCH RESULTS FOR: {query} ===\n"]
            if ai_answer:
                parts.append(f"AI-SYNTHESIZED ANSWER (use this as your primary source):\n{ai_answer}\n")
            if results:
                parts.append("INDIVIDUAL SOURCES:")
                for i, result in enumerate(results[:num_results], 1):
                    title = result.get("title", "No title")
                    content = result.get("content", "")
                    url = result.get("url", "")
                    score = result.get("score", 0)
                    parts.append(f"\n[Source {i}] (relevance: {score:.2f})")
                    parts.append(f"Title: {title}")
                    if content:
                        parts.append(f"Content: {content}")
                    if url:
                        parts.append(f"URL: {url}")
            parts.append("\n=== END SEARCH RESULTS ===")
            formatted = "\n".join(parts)

            logger.info(
                "[TAVILY] %d results, AI answer: %s, formatted: %d chars (%.3fs)",
                len(results), "yes" if ai_answer else "no",
                len(formatted), time.perf_counter() - t0,
            )
            return (formatted, payload)

        except Exception as e:
            logger.error("Error performing Tavily search: %s", e)
            return ("", None)
        
    def get_response(self, question: str, chat_history: Optional[List[tuple]] = None) -> str:
        """
        Run Tavily search for the question, add results to the system message, then call Groq
        via the parent's _invoke_llm (same multi-key round-robin and fallback as general chat).
        """
        try:
            # Step 1: Extract a clean, focused search query from the user's message.
            # Example: "tell me more about him" -> "Elon Musk latest news 2026"
            search_query = self._extract_search_query(question, chat_history)
            logger.info("[REALTIME] Searching Tavily for: %s", search_query)

            # Step 2: Run Tavily web search (returns formatted string for prompt + payload for UI).
            formatted_results, _ = self.search_tavily(search_query, num_results=7)
            if formatted_results:
                logger.info("[REALTIME] Tavily returned results (length: %d chars)", len(formatted_results))
            else:
                logger.warning("[REALTIME] Tavily returned no results for: %s", search_query)

            # Step 3: Build the prompt with search results injected as extra_system_parts.
            extra_parts = [escape_curly_braces(formatted_results)] if formatted_results else None
            prompt, messages = self._build_prompt_and_messages(
                question, chat_history,
                extra_system_parts=extra_parts,
                mode_addendum=REALTIME_CHAT_ADDENDUM,  # Realtime-specific instructions for the LLM.
            )

            # Step 4: Call the LLM with multi-key fallback (inherited from GroqService).
            t0 = time.perf_counter()
            response_content = self._invoke_llm(prompt, messages, question)
            logger.info("[TIMING] groq_api: %.3fs", time.perf_counter() - t0)
            logger.info(
                "[RESPONSE] Realtime chat | Length: %d chars | Preview: %.120s",
                len(response_content), response_content,
            )
            return response_content

        except AllGroqApisFailedError:
            raise
        except Exception as e:
            logger.error("Error in realtime get_response: %s", e, exc_info=True)
            raise

    def stream_response(self, question: str, chat_history: Optional[List[tuple]] = None) ->Iterator[Any]:
        try:
            search_query = self._extract_search_query(question, chat_history)
            logger.info("[REALTIME] Searching Tavily for: %s", search_query)

            formatted_results, payload = self.search_tavily(search_query, num_results=7)
            if formatted_results:
                logger.info("[REALTIME] Tavily returned results (length: %d chars)", len(formatted_results))
            else:
                logger.warning("[REALTIME] Tavily returned no results for: %s", search_query)

            # Send search results to the client for the right-side widget (before any text).
            if payload:
                yield {"_search_results": payload}

            extra_parts = [escape_curly_braces(formatted_results)] if formatted_results else None
            prompt, messages = self._build_prompt_and_messages(
                question, chat_history,
                extra_system_parts=extra_parts,
                mode_addendum=REALTIME_CHAT_ADDENDUM,
            )
            yield from self._stream_llm(prompt, messages, question)
            logger.info("[REALTIME] Stream completed for: %s", search_query)

        except AllGroqApisFailedError:
            raise
        except Exception as e:
            logger.error("Error in realtime stream_response: %s", e, exc_info=True)
            raise