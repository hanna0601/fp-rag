from __future__ import annotations

from typing import Any

import httpx

from app.config import settings


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
        payload = {
            "model": settings.embed_model,
            "input": texts,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        return [item["embedding"] for item in data["data"]]

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
            response.raise_for_status()
            data = response.json()
        message = data["choices"][0]["message"]["content"]
        if isinstance(message, list):
            return "".join(part.get("text", "") for part in message if part.get("type") == "text")
        return str(message)
