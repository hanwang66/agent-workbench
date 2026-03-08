from __future__ import annotations

import asyncio
from typing import Any
import httpx


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        timeout_seconds: int,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._timeout = timeout_seconds
        self._max_retries = max(0, max_retries)
        self._retry_backoff_seconds = max(0.0, retry_backoff_seconds)

    @property
    def model(self) -> str:
        return self._model

    async def _post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:
        attempt = 0
        while True:
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("Ollama response is not a JSON object")
                return data
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                # Retry transient upstream failures only.
                if status < 500 or attempt >= self._max_retries:
                    raise
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt >= self._max_retries:
                    raise

            attempt += 1
            await asyncio.sleep(self._retry_backoff_seconds * attempt)

    async def generate(self, prompt: str) -> str:
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self._temperature},
        }
        data = await self._post_json(url=url, payload=payload)

        if "response" not in data:
            raise ValueError("Ollama response missing 'response' field")

        return str(data["response"]).strip()

    async def embed(self, text: str, embed_model: str) -> list[float]:
        url = f"{self._base_url}/api/embeddings"
        payload = {"model": embed_model, "prompt": text}
        data = await self._post_json(url=url, payload=payload)

        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise ValueError("Ollama embeddings response missing 'embedding' field")

        return [float(value) for value in embedding]

    async def list_models(self) -> list[str]:
        url = f"{self._base_url}/api/tags"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

        if not isinstance(data, dict):
            raise ValueError("Ollama tags response is invalid")

        models = data.get("models", [])
        if not isinstance(models, list):
            return []

        names: list[str] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("model")
            if isinstance(name, str) and name:
                names.append(name)

        return names

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self._temperature},
        }
        if tools:
            payload["tools"] = tools

        data = await self._post_json(url=url, payload=payload)
        message = data.get("message")
        if not isinstance(message, dict):
            raise ValueError("Ollama chat response missing 'message' field")
        return message
