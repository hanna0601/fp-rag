from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import settings


class MistralAPIError(RuntimeError):
    pass


class MistralClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or settings.mistral_api_key
        self.base_url = base_url or settings.mistral_api_base

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("MISTRAL_API_KEY is not configured.")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        async with httpx.AsyncClient(timeout=60.0) as client:
            for start in range(0, len(texts), settings.embed_batch_size):
                batch = texts[start : start + settings.embed_batch_size]
                payload = {
                    "model": settings.embed_model,
                    "input": batch,
                }
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers=self._headers(),
                    json=payload,
                )
                data = self._handle_response(response, "embeddings")
                embeddings.extend(item["embedding"] for item in data["data"])
        return embeddings

    async def chat(self, messages: list[dict[str, Any]], temperature: float = 0.1) -> str:
        payload = {
            "model": settings.chat_model,
            "temperature": temperature,
            "messages": messages,
            "response_format": {"type": "text"},
        }
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            data = self._handle_response(response, "chat completions")
        message = data["choices"][0]["message"]["content"]
        if isinstance(message, list):
            return "".join(part.get("text", "") for part in message if part.get("type") == "text")
        return str(message)

    def _handle_response(self, response: httpx.Response, operation: str) -> dict[str, Any]:
        if response.is_success:
            return response.json()

        detail = self._extract_error_detail(response)
        raise MistralAPIError(
            f"Mistral {operation} request failed with HTTP {response.status_code}: {detail}"
        )

    def _extract_error_detail(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
        except json.JSONDecodeError:
            return response.text.strip() or "No error body returned by Mistral."

        if isinstance(payload, dict):
            if "message" in payload:
                return str(payload["message"])
            if "error" in payload:
                error = payload["error"]
                if isinstance(error, dict):
                    if "message" in error:
                        return str(error["message"])
                    return json.dumps(error)
                return str(error)
            return json.dumps(payload)
        return str(payload)
