#!/usr/bin/env python3
"""
Test script to verify imports are working correctly
"""
import os
import sys

print("Current working directory:", os.getcwd())
print("Python path:", sys.path)

try:
    print("Attempting to import get_online_sentiment...")
    from utils.external_data import get_online_sentiment
    print("✅ Successfully imported get_online_sentiment")
except Exception as e:
    print("❌ Failed to import get_online_sentiment:", str(e))

try:
    print("Attempting to import get_online_sentiment_with_search...")
    from utils.external_data import get_online_sentiment_with_search
    print("✅ Successfully imported get_online_sentiment_with_search")
    print("Function path:", get_online_sentiment_with_search.__code__.co_filename)
except Exception as e:
    print("❌ Failed to import get_online_sentiment_with_search:", str(e))
    import traceback
    traceback.print_exc()

print("Test completed.") 