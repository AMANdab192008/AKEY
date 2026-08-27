from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from contextlib import asynccontextmanager
import json
import time
import re
import uvicorn
import logging
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor
import edge_tts
from app.models import ChatRequest, ChatResponse, TTSRequest

# User-friendly message when Groq rate limit (daily token quota) is exceeded.
RATE_LIMIT_MESSAGE = (
    "You've reached your daily API limit for this assistant. "
    "Your credits will reset in a few hours, or you can upgrade your plan for more. "
    "Please try again later."
)


def _is_rate_limit_error(exc: Exception) -> bool:
    """True if the exception is a Groq rate limit (429 / tokens per day.)"""
    msg = str(exc).lower()
    return "429" in str(exc) or "rate limit" in msg or "tokens per day" in msg


from app.services.vector_store import VectorStoreService
from app.services.groq_service import GroqService, AllGroqApisFailedError
from app.services.realtime_service import RealtimeGroqService
from app.services.chat_service import ChatService
from config import (
    VECTOR_STORE_DIR, GROQ_API_KEYS, GROQ_MODEL, TAVILY_API_KEY,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHAT_HISTORY_TURNS,
    ASSISTANT_NAME, TTS_VOICE, TTS_RATE,
)    

# ---------------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("A.K.E.Y")


# ---------------------------------------------------------------------------------
# GLOBAL SERVICE REFERENCES
# ---------------------------------------------------------------------------------
# set during startup (lifespan) and used by all route handlers.
# stored as globals so async endpoints can access the same service instances.
vector_store_service: VectorStoreService = None
groq_service: GroqService = None
realtime_service: RealtimeGroqService = None
chat_service: ChatService = None


def print_title():
    """Print the A.K.E.Y ASCII art banner to the console when the server starts."""
    title = """
    
     ___________________________________________________________________
    |                                                                   |
    |               █████╗ ██╗  ██╗███████╗██╗   ██╗                    |
    |              ██╔══██╗██║ ██╔╝██╔════╝╚██╗ ██╔╝                    |
    |              ███████║█████═╝ █████╗   ╚████╔╝                     |
    |              ██╔══██║██╔═██╗ ██╔══╝    ╚██╔╝                      |
    |              ██║  ██║██║  ██╗███████╗   ██║                       |
    |              ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝                       |
    |                                                                   |
    |           Automated Knowledge & Execution Yield                   |
    |                                                                   |
    |___________________________________________________________________|
    
    """
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(title)


# -----------------------------------------------------------------------------------
# LIFESPAIN (STARTUP / SHUTDOWN)
# -----------------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):

    global vector_store_service, groq_service, realtime_service, chat_service

    print_title()
    logger.info("=" * 60)
    logger.info("A.K.E.Y - Starting Up...")
    logger.info("=" * 60)
    logger.info("[CONFIG] Assistant name: %s", ASSISTANT_NAME)
    logger.info("[CONFIG] Groq model: %s", GROQ_MODEL)
    logger.info("[CONFIG] Groq API keys loaded: %d", len(GROQ_API_KEYS))
    logger.info("[CONFIG] Tavily API key: %s", "configured" if TAVILY_API_KEY else "NOT SET")
    logger.info("[CONFIG] Embedding model: %s", EMBEDDING_MODEL)
    logger.info("[CONFIG] Chunk size: %d | Overlap: %d | Max history turns: %d",
                CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHAT_HISTORY_TURNS)

    try:
        # Initialize vector store service
        logger.info("Initializing vector store service...")
        t0 = time.perf_counter()
        vector_store_service = VectorStoreService()
        vector_store_service.create_vector_store()
        logger.info("[TIMING] startup_vector_store: %.3fs", time.perf_counter() - t0)

        # Initialize Groq servive (general chat)
        logger.info("Initializing Groq service (general queries)...")
        groq_service = GroqService(vector_store_service)
        logger.info("Groq service initialized successfully")

        # Initialized Realtime Groq service (with Tavily search)
        logger.info("Initializing Realtime Groq service (with Tavily search)...")
        realtime_service = RealtimeGroqService(vector_store_service)
        logger.info("Realtime Groq service initialized successfully")

        # Initialize chat service
        logger.info("Initializing chat service...")
        chat_service = ChatService(groq_service, realtime_service)
        logger.info("Chat service initialized successfully")

        # Startup complete
        logger.info("=" * 60)
        logger.info("Service Status:")
        logger.info(" - Vector Store: Ready")
        logger.info(" - Groq AI (General): Ready")
        logger.info(" - Groq AI (Realtime): Ready")
        logger.info(" - Chat Service: Ready")
        logger.info("=" * 60)
        logger.info("A.K.E.Y is online and ready!")
        logger.info("API: http://localhost:8000")
        logger.info("Frontend: http://localhost:8000/app/ (open in browser)")
        logger.info("=" * 60)

        yield

        # Shutdown: Save active sessions
        logger.info("\nShutting down A.K.E.Y...")
        if chat_service:
            for session_id in list(chat_service.sessions.keys()):
                chat_service.save_chat_session(session_id)
        logger.info("All sessions saved. Goodbye!")

    except Exception as e:
        logger.error(f"Fatal error during startup: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------------
# FASTAPI APP AND CORS
# ---------------------------------------------------------------------------------
# lifespan runs once at startup (build services) and once at shutdown (save sessions).
app = FastAPI(
    title="A.K.E.Y API",
    description="Just A Rather Very Intelligent System",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

# Allow any origin so a frontend on another port or device can call this API without CORS errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TimingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - t0
        path = request.url.path
        logger.info("[REQUEST] %s %s -> %s (%.3fs)", request.method, path, response.status_code, elapsed)
        return response

app.add_middleware(TimingMiddleware)

@app.get("/api")
async def api_info():
    """Return the API name and a short description of each endpoint (for discovery)."""
    return {
        "message": "A.K.E.Y API",
        "endpoints": {
            "/chat": "General chat (non-streaming)",
            "/chat/stream": "General chat (streaming chunks)",
            "/chat/realtime": "Realtime chat (non-streaming)",
            "/chat/realtime/stream": "Realtime chat (streaming chunks)",
            "/chat/history/{session_id}": "Get chat history",
            "/health": "System health check",
            "/tts": "Text-to-speech (POST text, returns streamed MP3)"
        }
    }


@app.get("/health")
async def health():
    """Return 'healthy' and whether each service (vector_store, groq, realtime, chat) is initialized,"""
    return {
        "status": "healthy",
        "vector_store": vector_store_service is not None,
        "groq_service": groq_service is not None,
        "realtime_service": realtime_service is not None,
        "chat_service": chat_service is not None
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")

    logger.info("[API /chat] Incoming | session_id=%s | message_len=%d | message=%.100s",
                request.session_id or "new", len(request.message), request.message)

    try:
        session_id = chat_service.get_or_create_session(request.session_id)
        response_text = chat_service.process_message(session_id, request.message)
        chat_service.save_chat_session(session_id)
        logger.info("[API /chat] Done | session_id=%s | response_len=%d", session_id[:12], len(response_text))
        return ChatResponse(response=response_text, session_id=session_id)
    except ValueError as e:
        # Invalid session_id (e.g. path traversal ".." or too long).
        logger.warning("[API /chat] Invalid session_id: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    except AllGroqApisFailedError as e:
        logger.error("[API /chat] All Groq APIs failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        if _is_rate_limit_error(e):
            logger.warning("[API /chat] Rate limit hit: %s", 3)
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
        logger.error("[API /chat] Error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat:{str(e)}")

_SPLIT_RE = re.compile(r"(?<=[.!?,;:])\s+")
_MIN_WORDS_FIRST = 2
_MIN_WORDS = 3
_MERGE_IF_WORDS = 2

def _split_sentences(buf: str):

    parts = _SPLIT_RE.split(buf)

    if len(parts) <= 1:
        return [], buf

    raw = [p.strip() for p in parts[:-1] if p.strip()]
    sentences, pending = [], ""

    for s in raw:
        if pending:
            s = (pending + " " + s).strip()
            pending = ""
        min_req = _MIN_WORDS_FIRST if not sentences else _MIN_WORDS
        if len(s.split()) < min_req:
            pending = s
            continue
        sentences.append(s)
    remaining = (pending + " " + parts[-1].strip()).strip() if pending else parts[-1].strip()
    return sentences, remaining


def _merge_short(sentences):
    if not sentences:
        return []
    merged, i = [], 0
    while i < len(sentences):
        cur = sentences[i]
        j = i + 1
        while j < len(sentences) and len(sentences[j].split()) <= _MERGE_IF_WORDS:
            cur = (cur + " " + sentences[j]).strip()
            j += 1
        merged.append(cur)
        i = j
    return merged

def _generate_tts_sync(text: str, voice: str, rate: str) -> bytes:
    async def _inner():
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
        parts = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                parts.append(chunk["data"])
        return b"".join(parts)
    return asyncio.run(_inner())

_tts_pool = ThreadPoolExecutor(max_workers=4)

def _stream_generator(session_id: str, chunk_iter, is_realtime: bool, tts_enabled: bool = False):

    yield f"data: {json.dumps({'session_id': session_id, 'chunk': '', 'done': False})}\n\n"

    buffer = ""
    held = None
    is_first = True
    audio_queue = []

    def _submit(text):
        audio_queue.append((_tts_pool.submit(_generate_tts_sync, text, TTS_VOICE, TTS_RATE), text))

    def _drain_ready():
        events = []
        while audio_queue and audio_queue[0][0].done():
            fut, sent = audio_queue.pop(0)
            try:
                audio = fut.result()
                b64 = base64.b64encode(audio).decode("ascii")
                events.append(f"data: {json.dumps({'audio': b64, 'sentence': sent})}\n\n")
            except Exception as exc:
                logger.warning("[TTS-INLINE] Failed for '%s': %s", sent[:40], exc)
        return events

    try:

        for chunk in chunk_iter:

            if isinstance(chunk, dict) and "_search_results" in chunk:
                yield f"data: {json.dumps({'search_results': chunk['_search_results']})}\n\n"
                continue
            if not chunk:
                continue

            yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"

            if not tts_enabled:
                continue

            for ev in _drain_ready():
                yield ev

            buffer += chunk
            sentences, buffer = _split_sentences(buffer)
            sentences = _merge_short(sentences)

            if held and sentences and len(sentences[0].split()) <= _MERGE_IF_WORDS:
                held = (held + " " + sentences[0]).strip()
                sentences = sentences[1:]

            for i, sent in enumerate(sentences):

                min_w = _MIN_WORDS_FIRST if is_first else _MIN_WORDS

                if len(sent.split()) < min_w:
                    continue

                is_last = (i == len(sentences) - 1)

                if held:
                    _submit(held)
                    held = None
                    is_first = False

                if is_last:
                    held = sent

                else:
                    _submit(sent)
                    is_first = False

    except Exception as e:

        for fut, _ in audio_queue:
            fut.cancel()

        yield f"data: {json.dumps({'chunk': '', 'done': True, 'error': str(e)})}\n\n"
        return

    if tts_enabled:
        remaining = buffer.strip()

        if held:

            if remaining and len(remaining.split()) <= _MERGE_IF_WORDS:
                _submit((held + " " + remaining).strip())

            else:
                _submit(held)

                if remaining:
                    _submit(remaining)

        elif remaining:
            _submit(remaining)

        for fut, sent in audio_queue:

            try:
                audio = fut.result(timeout=15)
                b64 = base64.b64encode(audio).decode("ascii")
                yield f"data: {json.dumps({'audio': b64, 'sentence': sent})}\n\n"

            except Exception as exc:
                logger.warning("[TTS-INLINE] Failed for '%s': %s", sent[:40], exc)

    yield f"data: {json.dumps({'chunk': '', 'done': True, 'session_id': session_id})}\n\n"

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):

    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
    logger.info("[API /chat/stream] Incoming | session_id=%s | message_len=%d | message=%.100s",
                request.session_id or "new", len(request.message), request.message)

    try:
        session_id = chat_service.get_or_create_session(request.session_id)
        chunk_iter = chat_service.process_message_stream(session_id, request.message)
        return StreamingResponse(
            _stream_generator(session_id, chunk_iter, is_realtime=False, tts_enabled=request.tts),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except AllGroqApisFailedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:

        if _is_rate_limit_error(e):
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)

        logger.error("[API /chat/stream] Error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/realtime", response_model=ChatResponse)
async def chat_realtime(request: ChatRequest):

    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")

    if not realtime_service:
        raise HTTPException(status_code=503, detail="Realtime service not initialized")

    logger.info("[API /chat/realtime] Incoming | session_id=%s | message_len=%d | message=%.100s",
                request.session_id or "new", len(request.message), request.message)

    try:
        session_id = chat_service.get_or_create_session(request.session_id)
        response_text = chat_service.process_realtime_message(session_id, request.message)
        chat_service.save_chat_session(session_id)
        logger.info("[API /chat/realtime] Done | session_id=%s | response_len=%d", session_id[:12], len(response_text))
        return ChatResponse(response=response_text, session_id=session_id)

    except ValueError as e:
        logger.warning("[API /chat/realtime] Invalid session_id: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    except AllGroqApisFailedError as e:
        logger.error("[API /chat/realtime] All Groq APIs failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:

        if _is_rate_limit_error(e):
            logger.warning("[API /chat/realtime] Rate limit hit: %s", e)
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)

        logger.error("[API /chat/realtime] Error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/chat/realtime/stream")
async def chat_realtime_stream(request: ChatRequest):

    if not chat_service or not realtime_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    logger.info("[API /chat/realtime/stream] Incoming | session_id=%s | message_len=%d | message=%.100s",
                request.session_id or "new", len(request.message), request.message)

    try:
        session_id = chat_service.get_or_create_session(request.session_id)
        chunk_iter = chat_service.process_realtime_message_stream(session_id, request.message)
        return StreamingResponse(
            _stream_generator(session_id, chunk_iter, is_realtime=True, tts_enabled=request.tts),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except AllGroqApisFailedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:

        if _is_rate_limit_error(e):
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)

        logger.error("[API /chat/realtime/stream] Error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")

    try:
        #Returns in-memory messages for this session (empty if session not loaded).
        messages = chat_service.get_chat_history(session_id)
        return {
            "session_id": session_id,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages]
        }
    except Exception as e:
        logger.error(f"Error retrieving history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

@app.post("/tts")
async def text_to_speech(request: TTSRequest):

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    async def generate():

        try:
            communicate = edge_tts.Communicate(text=text, voice=TTS_VOICE, rate=TTS_RATE)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]

        except Exception as e:
            logger.error("[TTS] Error generating speech: %s", e)

    return StreamingResponse(
        generate(),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-cache"},
    )

_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/app/", status_code=302)


def run():

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    run()