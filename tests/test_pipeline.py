import pytest
from fastapi.testclient import TestClient

from app import ingest, vector_store
from app.main import app
from app.rag import NOT_FOUND, retrieve


@pytest.fixture(scope="module", autouse=True)
def indexed():
    n = ingest.ingest(recreate=True)
    assert n > 0
    yield n
    vector_store.reset_client()


client = TestClient(app)


def test_chunker_respects_size_and_overlap():
    text = "\n".join(f"Sentence number {i} about leave." for i in range(30))
    chunks = ingest.chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 + 40 for c in chunks)
    assert chunks[1][:15] in chunks[0]          # overlap carried forward


def test_reingest_is_idempotent(indexed):
    assert ingest.ingest(recreate=False) == indexed
    assert vector_store.count() == indexed       # same ids -> overwritten, not duplicated


@pytest.mark.parametrize("question,expected", [
    ("How many annual leave days are available?", "leave_policy.txt"),
    ("How many vacation days do I get?", "leave_policy.txt"),
    ("How many days can I work from home?", "remote_work_policy.md"),
    ("What is the hotel limit for London?", "expense_policy.txt"),
    ("How often must I change my password?", "it_security_policy.txt"),
    ("Does the company pay for my health insurance?", "benefits.md"),
])
def test_retrieval_finds_right_file(question, expected):
    hits = retrieve(question, top_k=3)
    assert expected in [h.source for h in hits]


def test_ask_returns_answer_and_sources():
    r = client.post("/ask", json={"question": "How many annual leave days do employees get?"})
    assert r.status_code == 200
    body = r.json()
    assert "20 days" in body["answer"]
    assert body["sources"][0]["file"] == "leave_policy.txt"


@pytest.mark.parametrize("question", [
    "Who won the football world cup?",
    "What is the capital of France?",
])
def test_unrelated_questions_are_refused(question):
    body = client.post("/ask", json={"question": question}).json()
    assert body["answer"] == NOT_FOUND
    assert body["sources"] == [] and body["mode"] == "no_context"


def test_validation_rejects_empty_question():
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_health():
    assert client.get("/health").json() == {"status": "ok", "indexed": True}


def test_root_serves_frontend():
    r = client.get("/")
    assert r.status_code == 200
    assert "Business RAG Assistant" in r.text
    assert "<form id=\"askForm\"" in r.text


def test_llm_path_builds_correct_request(monkeypatch):
    """No real API call: we replace the Anthropic client and inspect what we would send."""
    import types
    import anthropic
    from app import rag

    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            block = types.SimpleNamespace(type="text", text="Employees get 20 days.")
            return types.SimpleNamespace(content=[block])

    class FakeClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    body = rag.answer_question("How many annual leave days do employees get?")
    assert body["mode"] == "llm" and body["answer"] == "Employees get 20 days."
    assert captured["api_key"] == "test-key"
    assert "ONLY the supplied context" in captured["system"]
    user_msg = captured["messages"][0]["content"]
    assert "Context:" in user_msg and "leave_policy.txt" in user_msg
    assert "How many annual leave days" in user_msg


def test_gemini_path_builds_correct_request(monkeypatch):
    """No real API call: we replace the Google GenAI client and inspect what we would send."""
    import types
    from google import genai
    from app import rag

    captured = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(text="Employees get 20 days.")

    class FakeClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash")

    body = rag.answer_question("How many annual leave days do employees get?")
    assert body["mode"] == "llm" and body["answer"] == "Employees get 20 days."
    assert captured["api_key"] == "test-gemini-key"
    assert captured["model"] == "gemini-2.5-flash"
    assert "ONLY the supplied context" in captured["config"].system_instruction
    assert "Context:" in captured["contents"] and "leave_policy.txt" in captured["contents"]
    assert "How many annual leave days" in captured["contents"]
