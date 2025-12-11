"""Startup script for the AI Fake News Detector API."""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run the app
import uvicorn

if __name__ == "__main__":
    # Use PORT from environment (for cloud deployment) or default to 8000
    port = int(os.getenv('PORT', 8000))

    # Detect if running in production (Render sets PORT env var)
    is_production = os.getenv('PORT') is not None

    print("Starting AI Fake News Detector API...")
    print(f"Project root: {project_root}")
    print(f"Mode: {'Production' if is_production else 'Development'}")
    print(f"API will be available at: http://localhost:{port}")
    print(f"API docs at: http://localhost:{port}/docs")
    print("-" * 60)

    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=port,
        reload=not is_production,  # Disable reload in production
        timeout_keep_alive=120  # Longer timeout for model loading
    )
