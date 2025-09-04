#!/usr/bin/env python3
"""
Startup script for the Knowledge Assistant.
This script can be used to run the application locally without Docker.
"""
import os
import sys
import argparse
import uvicorn
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import config


def main():
    """Main function to run the Knowledge Assistant."""
    parser = argparse.ArgumentParser(description="Knowledge Assistant Startup Script")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    # Validate configuration
    try:
        config.validate()
        print("✓ Configuration validated")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        print("Please ensure OPENAI_API_KEY is set in your .env file")
        sys.exit(1)
    
    # Check if documents exist
    docs_path = Path(config.DOCS_PATH)
    if not docs_path.exists() or not list(docs_path.glob("*.md")):
        print(f"✗ No documents found in {docs_path}")
        print("Please ensure documentation files exist in the data/docs directory")
        sys.exit(1)
    
    print(f"✓ Found documentation in {docs_path}")
    print(f"Starting Knowledge Assistant on {args.host}:{args.port}")
    
    # Run the application
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
