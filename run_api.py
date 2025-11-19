"""Startup script for the AI Fake News Detector API."""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run the app
import uvicorn

if __name__ == "__main__":
    print("Starting AI Fake News Detector API...")
    print(f"Project root: {project_root}")
    print("API will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("-" * 60)

    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
