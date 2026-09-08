import atexit
import os
import random
import threading
import time
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from openai import APIConnectionError, APIStatusError, OpenAI
from pydantic import Field

# Process resources stay outside checkpoint state. One connection pool serves
# all agents and broadcast workers using the same endpoint and credential.
_resources = {}
_resource_lock = threading.Lock()


def _client_resources(url, api_key, timeout, concurrency):
    key = (os.getpid(), url, api_key, timeout, concurrency)
    with _resource_lock:
        if key not in _resources:
            _resources[key] = (
                OpenAI(api_key=api_key, base_url=url, timeout=timeout, max_retries=0),
                threading.BoundedSemaphore(concurrency),
            )
        return _resources[key]


@atexit.register
def close_clients():
    with _resource_lock:
        for client, _ in _resources.values():
            client.close()
        _resources.clear()


class CustomLLM(LLM):
    max_token: int
    URL: str = "http://xxxxx"
    api_key: str = Field(default="", repr=False, exclude=True)
    max_retries: int = Field(default=2, ge=0)
    request_timeout: float = Field(default=60.0, gt=0)
    max_concurrency: int = Field(default=40, ge=1)
    retry_min_seconds: float = 0.5
    retry_max_seconds: float = 8.0
    logger: Any
    model: str
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.8, gt=0, le=1)
    top_k: int = Field(default=20, ge=0)
    enable_thinking: bool = False

    @property
    def _llm_type(self) -> str:
        return "CustomLLM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        history: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        client, slots = _client_resources(
            self.URL, self.api_key, self.request_timeout, self.max_concurrency
        )
        started = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                with slots:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        stop=stop,
                        n=1,
                        max_tokens=self.max_token,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        extra_body={"enable_thinking": self.enable_thinking, "top_k": self.top_k},
                    )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("LLM response has no text content")
                self.logger.debug(
                    "LLM completed seconds=%.3f attempts=%d",
                    time.perf_counter() - started,
                    attempt + 1,
                )
                return content.strip()
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                transient = isinstance(exc, APIConnectionError) or (
                    isinstance(exc, APIStatusError) and (status in (408, 409, 429) or status >= 500)
                )
                if not transient or attempt >= self.max_retries:
                    self.logger.error(
                        "LLM failed type=%s status=%s attempts=%d seconds=%.3f",
                        type(exc).__name__,
                        status,
                        attempt + 1,
                        time.perf_counter() - started,
                    )
                    raise
                wait_seconds = min(self.retry_min_seconds * 2**attempt, self.retry_max_seconds) * (
                    1 + random.SystemRandom().random() * 0.2
                )
                self.logger.warning(
                    "LLM retry type=%s status=%s retry=%d/%d wait=%.2fs",
                    type(exc).__name__,
                    status,
                    attempt + 1,
                    self.max_retries,
                    wait_seconds,
                )
                time.sleep(wait_seconds)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "enable_thinking": self.enable_thinking,
            "max_token": self.max_token,
            "URL": self.URL,
            "max_retries": self.max_retries,
            "request_timeout": self.request_timeout,
            "max_concurrency": self.max_concurrency,
        }
