"""
Standalone sanity check: confirms your .env file's GEMINI_API_KEY loads
correctly and that a basic Gemini API call returns a response, BEFORE
wiring anything into the full RAG pipeline.

Usage:
    pip install google-generativeai python-dotenv
    python test_gemini_key.py
"""

import os
import sys

from dotenv import load_dotenv
import google.generativeai as genai

MODEL_NAME = "gemini-3.6-flash"


def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found. Checklist:")
        print("  1. Does a .env file exist in this folder?")
        print("  2. Does it contain a line like: GEMINI_API_KEY=your-key-here")
        print("  3. Are you running this script from the same folder as .env?")
        sys.exit(1)

    print(f"Found GEMINI_API_KEY (starts with: {api_key[:6]}...)")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    print(f"Sending a test prompt to {MODEL_NAME}...")
    try:
        response = model.generate_content("Say hello in exactly one short sentence.")
    except Exception as e:
        print(f"API call failed: {e}")
        print("Common causes: invalid/expired key, no network, or free-tier rate limit hit.")
        sys.exit(1)

    print("\nSuccess! Response from Gemini:")
    print(response.text)


if __name__ == "__main__":
    main()