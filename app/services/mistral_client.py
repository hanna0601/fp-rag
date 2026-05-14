from __future__ import annotations

import json                  # parse error response bodies from Mistral
from typing import Any       # type hint for flexible message/response dicts

import httpx                 # async HTTP client — used instead of requests because FastAPI is async

from app.config import settings  # api_key, base_url, model names, batch size, timeouts


# Custom exception so callers can catch Mistral errors specifically
# instead of catching the generic Exception — makes error handling more precise
class MistralAPIError(RuntimeError):
    pass


class MistralClient:
    """Thin async HTTP wrapper around the Mistral REST API.

    Two public methods:
      embed_texts() — converts text into float vectors (used for semantic search)
      chat()        — sends messages to the LLM and returns the text response

    No Mistral SDK used — raw httpx calls give full visibility and control
    over request/response handling, batching, and error parsing.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        # allow override for testing (pass a mock key/url) without changing settings
        self.api_key = api_key or settings.mistral_api_key
        self.base_url = base_url or settings.mistral_api_base

    def _headers(self) -> dict[str, str]:
        # called before every request — raises immediately if key is missing
        # rather than getting a cryptic 401 from the API
        if not self.api_key:
            raise RuntimeError("MISTRAL_API_KEY is not configured.")
        return {
            "Authorization": f"Bearer {self.api_key}",  # Mistral uses Bearer token auth
            "Content-Type": "application/json",          # we send JSON bodies
            "Accept": "application/json",                # we expect JSON back
        }

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert a list of text strings into embedding vectors.

        Returns list[list[float]] — one float vector per input text.
        Vectors are 1024-dimensional for mistral-embed.

        Why batch?
        The Mistral API accepts multiple texts in one request. Sending all
        texts at once in one call is much faster than one request per text.
        batch_size=16 is conservative — large enough to be efficient, small
        enough to avoid hitting payload size limits or timeouts.

        Example:
          input:  ["The attention mechanism...", "BERT is designed to..."]
          output: [[0.12, -0.34, 0.87, ...], [0.09, 0.44, -0.21, ...]]
        """
        if not texts:
            return []   # nothing to embed — return early, avoid an empty API call
        embeddings: list[list[float]] = []
        async with httpx.AsyncClient(timeout=60.0) as client:   # 60s timeout per batch request
            for start in range(0, len(texts), settings.embed_batch_size):
                # slice a batch of up to embed_batch_size texts
                # e.g. texts[0:16], texts[16:32], texts[32:48] ...
                batch = texts[start : start + settings.embed_batch_size]
                payload = {
                    "model": settings.embed_model,  # "mistral-embed"
                    "input": batch,                 # list of strings to embed
                }
                response = await client.post(
                    f"{self.base_url}/embeddings",  # POST https://api.mistral.ai/v1/embeddings
                    headers=self._headers(),
                    json=payload,
                )
                data = self._handle_response(response, "embeddings")
                # Mistral response shape: {"data": [{"embedding": [0.12, ...]}, ...]}
                # data["data"] is a list — one item per input text in the batch
                embeddings.extend(item["embedding"] for item in data["data"])
        return embeddings  # full list, same order as input texts

    async def chat(self, messages: list[dict[str, Any]], temperature: float = 0.1) -> str:
        """Send a conversation to the LLM and return its text reply.

        messages format (OpenAI-compatible):
          [
            {"role": "system", "content": "You are a RAG assistant..."},
            {"role": "user",   "content": "Question: ... Evidence: ..."}
          ]

        temperature controls randomness:
          0.0 = fully deterministic (used for HyDE hypothetical generation)
          0.1 = near-deterministic (used for final answer — consistent but not robotic)
          1.0 = creative/random

        timeout=90s because LLM generation is slower than embedding —
        generating a full answer over 8 evidence chunks can take several seconds.
        """
        payload = {
            "model": settings.chat_model,              # "mistral-small-latest"
            "temperature": temperature,                 # how random the output is
            "messages": messages,                       # full conversation history
            "response_format": {"type": "text"},        # plain text, not JSON mode
        }
        async with httpx.AsyncClient(timeout=90.0) as client:   # longer timeout for generation
            response = await client.post(
                f"{self.base_url}/chat/completions",    # POST https://api.mistral.ai/v1/chat/completions
                headers=self._headers(),
                json=payload,
            )
            data = self._handle_response(response, "chat completions")

        # Mistral response shape: {"choices": [{"message": {"content": "..."}}]}
        # choices[0] = first (and only) completion
        # message["content"] = the LLM's reply — usually a plain string
        message = data["choices"][0]["message"]["content"]

        # defensive: some Mistral models return content as a list of parts
        # e.g. [{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}]
        # join all text parts into one string in that case
        if isinstance(message, list):
            return "".join(part.get("text", "") for part in message if part.get("type") == "text")
        return str(message)  # normal case: content is already a plain string

    def _handle_response(self, response: httpx.Response, operation: str) -> dict[str, Any]:
        """Return parsed JSON if the request succeeded, otherwise raise MistralAPIError.

        is_success = HTTP status 200-299
        Anything else (400, 401, 429, 500 etc.) → extract the error detail and raise.

        Raising MistralAPIError (not generic Exception) lets callers catch it specifically:
          except MistralAPIError: → handle API failure
          except Exception:       → handle unexpected errors
        """
        if response.is_success:
            return response.json()   # parse and return the JSON body

        # request failed — extract a human-readable error message before raising
        detail = self._extract_error_detail(response)
        raise MistralAPIError(
            f"Mistral {operation} request failed with HTTP {response.status_code}: {detail}"
        )

    def _extract_error_detail(self, response: httpx.Response) -> str:
        """Pull a readable error message out of Mistral's error response body.

        Mistral returns errors in several different shapes depending on the error type.
        This method handles all of them so the raised exception always has a clear message.

        Shape 1 — top-level message:
          {"message": "Invalid API key"}
          → returns "Invalid API key"

        Shape 2 — nested error object:
          {"error": {"message": "Rate limit exceeded", "type": "rate_limit"}}
          → returns "Rate limit exceeded"

        Shape 3 — error is a plain string:
          {"error": "model not found"}
          → returns "model not found"

        Shape 4 — unexpected JSON structure:
          {"something": "unexpected"}
          → returns the full JSON as a string

        Shape 5 — response body is not JSON at all (e.g. plain text HTML error page):
          "Internal Server Error"
          → returns that raw text
        """
        try:
            payload = response.json()   # attempt to parse the body as JSON
        except json.JSONDecodeError:
            # body is not JSON — return raw text (e.g. an HTML 502 error page)
            return response.text.strip() or "No error body returned by Mistral."

        if isinstance(payload, dict):
            if "message" in payload:              # Shape 1 — top-level message key
                return str(payload["message"])
            if "error" in payload:
                error = payload["error"]
                if isinstance(error, dict):
                    if "message" in error:        # Shape 2 — nested error.message
                        return str(error["message"])
                    return json.dumps(error)       # Shape 2 fallback — dump full error dict
                return str(error)                  # Shape 3 — error is a plain string
            return json.dumps(payload)             # Shape 4 — unknown structure, dump all
        return str(payload)                        # Shape 5 — body parsed but isn't a dict
