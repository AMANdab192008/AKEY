"""
RUN SCRIPT - Start the A.K.E.Y server
=====================================

PURPOSE:
  Single entry point to start the backend. Run this once per user/machine;
  the server then handles all chat and realtime requests for that instance.

WHAT IT DOES:
  - Imports the FastAPI app from app.main.
  - Runs it with uvicorn on host 0.0.0.0 (accept connections from any interface) and port 8000.
  - reload=True means any change to Python files will restart the server (handy for development).

USAGE:
  python run.py

Then open http://localhost:8000 in the browser, or use the API from another app.

NOTE:
  Before running, set GROQ_API_KEY (and optionally TAVILY_API_KEY for realtime search) in .env.

--------------------------------------------------------------------------------
WHAT IS UVICORN?
--------------------------------------------------------------------------------
Uvicorn is an ASGI (Asynchronous Server Gateway Interface) server for Python.

In simple terms:
  - FastAPI defines *what* happens when a request arrives (your route handlers).
  - Uvicorn is the *engine* that actually listens on a network port, accepts
    incoming HTTP connections, and hands each request to FastAPI for processing.

Think of it like this:
  - FastAPI = the chef who knows every recipe.
  - Uvicorn = the restaurant that seats customers and delivers their orders to the chef.

ASGI (vs. WSGI):
  - WSGI (older standard): handles one request at a time per worker. Used by
    Flask, Django (traditional).
  - ASGI (newer standard): supports async/await, WebSockets, and can handle
    many concurrent connections efficiently. FastAPI is built for ASGI.

--------------------------------------------------------------------------------
THE if __name__ == "__main__": GUARD
--------------------------------------------------------------------------------
This is a standard Python idiom. When you run a file directly:
  python run.py
Python sets the special variable __name__ to "__main__" for that file.
  """

import uvicorn

# --------------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------------
# Only run uvicorn when this file is executed directly (python run.py),
# not when it is imported by another module.
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
    