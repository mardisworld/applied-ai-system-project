from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib import error, request

from src.env_loader import load_project_env
from src.retrieval import RetrievalContext


DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-5.4"
DEFAULT_TIMEOUT_SECONDS = 60


@dataclass
class LLMConfig:
    api_key: str
    base_url: str = DEFAULT_LLM_BASE_URL
    model: str = DEFAULT_LLM_MODEL
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> "LLMConfig":
        load_project_env()
        api_key = (os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise ValueError(
                "Missing LLM API key. Set LLM_API_KEY or OPENAI_API_KEY to enable grounded LLM recommendations."
            )

        base_url = (
            os.environ.get("LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_LLM_BASE_URL
        ).strip().rstrip("/")
        model = (os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODEL).strip()
        timeout_value = (os.environ.get("LLM_TIMEOUT_SECONDS") or "").strip()

        timeout_seconds = DEFAULT_TIMEOUT_SECONDS
        if timeout_value:
            try:
                timeout_seconds = int(timeout_value)
            except ValueError as exc:
                raise ValueError("LLM_TIMEOUT_SECONDS must be an integer.") from exc

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
        )


def llm_is_configured() -> bool:
    load_project_env()
    return bool((os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip())


def _extract_message_text(response_payload: Dict[str, Any]) -> str:
    choices = response_payload.get("choices") or []
    if not choices:
        raise RuntimeError("LLM response did not contain any choices.")

    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and part.get("text"):
                text_parts.append(str(part["text"]))
        return "\n".join(text_parts).strip()

    raise RuntimeError("LLM response did not contain text content.")


def send_chat_completion(
    prompt: str,
    config: Optional[LLMConfig] = None,
    system_prompt: Optional[str] = None,
) -> str:
    llm_config = config or LLMConfig.from_env()
    request_body = {
        "model": llm_config.model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt or "You are a precise music recommendation assistant. Ground every recommendation in the provided retrieval context.",
            },
            {"role": "user", "content": prompt},
        ],
    }

    payload = json.dumps(request_body).encode("utf-8")
    http_request = request.Request(
        f"{llm_config.base_url}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {llm_config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=llm_config.timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"LLM API request failed with HTTP {exc.code}: {response_body}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"LLM API request failed: {exc.reason}") from exc

    return _extract_message_text(response_payload)


def generate_grounded_recommendation_text(
    retrieval_context: RetrievalContext,
    recommendation_count: int = 5,
    config: Optional[LLMConfig] = None,
    system_prompt: Optional[str] = None,
) -> str:
    prompt = retrieval_context.to_llm_prompt(recommendation_count=recommendation_count)
    return send_chat_completion(prompt, config=config, system_prompt=system_prompt)