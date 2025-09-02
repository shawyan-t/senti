"""
Pre-warm model weights and caches during the Render build.

This script downloads and caches the transformers and sentence-transformers
artifacts used by the API so the first request doesn't incur large downloads.
It is safe to run multiple times; it will be a no-op if caches already exist.
"""
import os

def main():
    # Ensure huggingface caches land in a project directory that persists in the image
    cache = os.getenv("TRANSFORMERS_CACHE") or os.getenv("HF_HOME")
    if not cache:
        cache = os.path.join(os.path.dirname(__file__), "..", ".cache", "huggingface")
        cache = os.path.abspath(cache)
        os.environ["TRANSFORMERS_CACHE"] = cache
        os.environ["HF_HOME"] = cache

    os.makedirs(cache, exist_ok=True)

    try:
        # 1) Sentiment pipeline (twitter-roberta)
        from transformers import pipeline
        _ = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        )
    except Exception as e:
        print("[prewarm] transformers sentiment pipeline failed:", e)

    try:
        # 2) Emotion classification (distilroberta)
        from transformers import pipeline as p2
        _ = p2(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
        )
    except Exception as e:
        print("[prewarm] transformers emotion pipeline failed:", e)

    try:
        # 3) Sentence embeddings (all-MiniLM-L6-v2)
        from sentence_transformers import SentenceTransformer
        _ = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print("[prewarm] sentence-transformers prewarm failed:", e)

    print("[prewarm] Completed model cache warmup at:", cache)

if __name__ == "__main__":
    main()

