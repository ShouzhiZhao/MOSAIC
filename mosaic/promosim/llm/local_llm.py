import re
import threading
from typing import Any, List, Mapping, Optional

import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic import Field
from transformers import AutoModelForCausalLM, AutoTokenizer


class SingletonLocalLLM:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, config, logger, api_key, api_base) -> "LocalLLM":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = LocalLLM(
                        max_token=config["max_token"],
                        model_path=api_base,
                        logger=logger,
                        temperature=config.get("temperature", 0.7),
                        top_p=config.get("top_p", 0.8),
                        top_k=config.get("top_k", 20),
                        enable_thinking=config.get("enable_thinking", False),
                    )
        settings = {
            "max_token": config["max_token"],
            "temperature": config.get("temperature", 0.7),
            "top_p": config.get("top_p", 0.8),
            "top_k": config.get("top_k", 20),
            "enable_thinking": config.get("enable_thinking", False),
            "model_path": api_base,
        }
        if any(getattr(cls._instance, k) != v for k, v in settings.items()):
            raise ValueError("Local LLM settings changed; start a new simulator process")
        return cls._instance


class LocalLLM(LLM):
    model_path: str = "local"
    max_token: int
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.8, gt=0, le=1)
    top_k: int = Field(default=20, ge=0)
    enable_thinking: bool = False
    model_name: str = ""
    tokenizer: Any = None
    model: Any = None
    logger: Any = None

    def __init__(self, **data):
        super().__init__(**data)
        self.model_name = data["model_path"].split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(data["model_path"], trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if not torch.cuda.is_available():
            raise RuntimeError("PromoSim local LLM requires CUDA")
        torch_dtype = torch.float16
        device_map = "auto"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                data["model_path"],
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                data["model_path"],
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            ).to("cuda")
        self.model.eval()

    @property
    def _llm_type(self) -> str:
        return self.model_name

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        history: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        message = [{"role": "user", "content": prompt}]
        device = self.model.get_input_embeddings().weight.device
        # Based on Qwen3 official docs: enable_thinking should be passed to apply_chat_template
        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask
        prompt_len = len(input_ids[0])
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_token,
            do_sample=self.temperature > 0,
            num_beams=1,
            **(
                {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k}
                if self.temperature > 0
                else {}
            ),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_ids = output_ids[0][prompt_len:]
        response = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        response = self._strip_think(response)
        response = self._apply_stop(response, stop)
        return response

    def _apply_stop(self, text: str, stop: Optional[List[str]]) -> str:
        if not stop:
            return text
        earliest = None
        for s in stop:
            if not s:
                continue
            idx = text.find(s)
            if idx == -1:
                continue
            if earliest is None or idx < earliest:
                earliest = idx
        if earliest is None:
            return text
        return text[:earliest].rstrip()

    def _strip_think(self, text: str) -> str:
        text = re.sub(
            r"<\|begin_of_thought\|>[\s\S]*?<\|end_of_thought\|>", "", text, flags=re.IGNORECASE
        )
        for tag in [
            "think",
            "thought",
            "analysis",
            "assistant_thought",
            "assistant_analysis",
            "internal",
        ]:
            text = re.sub(rf"<{tag}>[\s\S]*?</{tag}>", "", text, flags=re.IGNORECASE)
            text = re.sub(rf"</?{tag}>", "", text, flags=re.IGNORECASE)
        return text.strip()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_token": self.max_token,
            "model Path": self.model_path,
        }
