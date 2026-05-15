from __future__ import annotations

# FastAPI core — the web framework that handles routing, request parsing, and response serialization
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
# CORS middleware — allows the browser UI (potentially on a different port/domain) to call the API
from fastapi.middleware.cors import CORSMiddleware
# HTMLResponse — tells FastAPI the endpoint returns raw HTML, not JSON
from fastapi.responses import HTMLResponse
# StaticFiles — mounts a directory so JS/CSS assets are served directly without a route handler
from fastapi.staticfiles import StaticFiles
# Jinja2Templates — renders server-side HTML templates with Python values injected
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.schemas import IngestionResponse, QueryRequest, QueryResponse
from app.services.mistral_client import MistralClient
from app.services.rag_service import RagService
from app.services.storage import Storage


# --- Module-level singletons (created once at startup, shared across all requests) ---
# Why singletons? Each of these is stateful or expensive to construct:
#   Storage opens a SQLite connection pool; recreating it per-request would thrash the disk.
#   MistralClient holds HTTP connection settings; recreating it per-request wastes setup time.
#   RagService holds the in-memory corpus cache (via HybridRetriever); per-request creation
#   would rebuild the full BM25 index from SQLite on every query — O(N) on every call.
# FastAPI runs in an async event loop, so these singletons are safe — there is no thread
# contention because async Python only runs one coroutine at a time per event loop.
storage = Storage(settings.db_path)
mistral_client = MistralClient()
rag_service = RagService(storage=storage, mistral_client=mistral_client)

# --- FastAPI application setup ---
# title and version appear in the auto-generated OpenAPI docs at /docs
app = FastAPI(title=settings.app_name, version="0.1.0")

# CORS (Cross-Origin Resource Sharing) middleware
# Browsers block JS from calling APIs on a different origin (protocol+host+port) by default.
# allow_origins=["*"] disables that restriction — acceptable in development because:
#   1. The app runs locally with no sensitive user data
#   2. There is no auth — anyone with the URL can already hit the API
# In production: restrict to specific origins, e.g. allow_origins=["https://myapp.example.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # accept requests from any origin
    allow_credentials=True,    # allow cookies and Authorization headers to be forwarded
    allow_methods=["*"],       # allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],       # allow any request headers (Content-Type, Authorization, etc.)
)

# Serve files from app/static/ at the URL prefix /static
# e.g. app/static/style.css → http://localhost:8000/static/style.css
# name="static" lets Jinja2 templates use url_for("static", path="style.css") to generate URLs
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Jinja2 template engine — looks for .html files in the app/templates/ directory
# Used by the index() route to render the full-page UI with server-injected variables
templates = Jinja2Templates(directory="app/templates")


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    # TemplateResponse renders index.html and injects the context dict as template variables
    # request is required by Jinja2 to build absolute URLs (e.g. for url_for())
    # app_name is injected so the template can display "First-Principles RAG" as the page title
    return templates.TemplateResponse("index.html", {"request": request, "app_name": settings.app_name})


@app.get("/health")
async def health() -> dict[str, str]:
    # Minimal liveness probe — used by Docker/k8s health checks and monitoring tools.
    # Returns 200 OK if the process is alive. Does NOT check DB connectivity or API keys
    # (a deep health check would hit the DB and return 503 on failure; we keep it simple here).
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestionResponse)
async def ingest(files: list[UploadFile] = File(...)) -> IngestionResponse:
    """Accept one or more PDF file uploads and index them into the knowledge base.

    File(...) means the field is required — FastAPI returns 422 Unprocessable Entity
    automatically if the client sends no files.

    Error mapping:
      ValueError → 400 Bad Request   (predictable user errors: non-PDF, empty file)
      Exception  → 500 Internal Error (unexpected failures: DB write error, API down)

    Why separate error types?
    400 tells the client "you did something wrong — fix your request."
    500 tells the client "something broke on our side — try again later."
    Mixing them (always returning 500) hides user errors and makes debugging harder.

    response_model=IngestionResponse causes FastAPI to validate and serialize the return
    value through Pydantic — extra fields are stripped, missing required fields raise 500.
    """
    try:
        results = await rag_service.ingest_files(files)
    except ValueError as exc:
        # Predictable client error: non-PDF upload, or PDF with no extractable text
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        # Unexpected server error: DB write failure, disk full, etc.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IngestionResponse(ingested=results)


@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest) -> QueryResponse:
    """Run a query against the indexed knowledge base and return a cited answer.

    payload is parsed from the JSON request body and validated by Pydantic
    before this function is called — FastAPI returns 422 if the body is malformed
    or violates field constraints (e.g. empty query string, top_k out of range).

    The full pipeline (intent → rewrite → embed → retrieve → generate → check)
    runs inside rag_service.answer_query() — this route is intentionally thin.
    """
    try:
        return await rag_service.answer_query(payload.query, payload.top_k)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/reset")
async def reset() -> dict[str, str]:
    """Delete all indexed documents and chunks, resetting the knowledge base to empty.

    Useful in development to start fresh without restarting the server.
    In production this would typically require authentication — left open here
    because the app is local-only and has no user accounts.

    Returns {"status": "ok"} on success, or raises 500 if the DB delete fails.
    """
    try:
        storage.reset()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "ok"}
