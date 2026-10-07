"""
Convenience entrypoint to launch the Kestrel Returns Risk API & TypeScript Frontend
----------------------------------------------------------------------------------
Usage:
    python run_service.py

Starts FastAPI + TypeScript frontend on http://localhost:8000
"""

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("🦅 Starting Kestrel Returns Risk Service...")
    print("📍 URL: http://localhost:8000")
    print("📍 API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
