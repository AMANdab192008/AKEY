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
API docs: http://localhost:8000/docs

NOTE:
  Before running, set GROQ_API_KEY (and optionally TAVILY_API_KEY for realtime search) in .env.
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
