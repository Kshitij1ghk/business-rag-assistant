import os
import tempfile

# Must be set BEFORE the app modules read their settings.
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["QDRANT_URL"] = ""
os.environ["QDRANT_PATH"] = tempfile.mkdtemp(prefix="qdrant_test_")
os.environ["LLM_API_KEY"] = ""          # forces the offline extractive mode
os.environ["MIN_SCORE"] = "0.18"   # calibrated for the offline hash embedder
