from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from .config import LLMConfig
from .image_io import load_rgb
from .query import QueryPlan, merge_llm_plan, static_expand

QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "semantic_zh": {"type": "string"},
        "semantic_en": {"type": "string"},
        "positive_tags": {"type": "array", "items": {"type": "string"}},
        "negative_tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["semantic_zh", "semantic_en", "positive_tags", "negative_tags"],
}


CAPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary_zh": {"type": "string"},
        "subjects": {"type": "array", "items": {"type": "string"}},
        "clothing": {"type": "array", "items": {"type": "string"}},
        "accessories": {"type": "array", "items": {"type": "string"}},
        "pose": {"type": "array", "items": {"type": "string"}},
        "composition": {"type": "array", "items": {"type": "string"}},
        "palette": {"type": "array", "items": {"type": "string"}},
        "style": {"type": "array", "items": {"type": "string"}},
        "mood": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "summary_zh",
        "subjects",
        "clothing",
        "accessories",
        "pose",
        "composition",
        "palette",
        "style",
        "mood",
    ],
}


class OllamaUnavailable(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}{endpoint}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaUnavailable(f"无法连接本地 Ollama：{exc}") from exc

    def is_available(self) -> bool:
        try:
            self.list_models()
            return True
        except OllamaUnavailable:
            return False

    def list_models(self) -> list[str]:
        try:
            request = urllib.request.Request(f"{self.config.base_url.rstrip('/')}/api/tags")
            with urllib.request.urlopen(request, timeout=2) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaUnavailable(f"无法连接本地 Ollama：{exc}") from exc
        models = data.get("models", []) if isinstance(data, dict) else []
        names: list[str] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "").strip()
            if name:
                names.append(name)
        return names

    @staticmethod
    def model_installed(wanted: str, installed: list[str]) -> bool:
        return any(item == wanted or item.startswith(wanted + "-") for item in installed)

    @staticmethod
    def _keep_alive_value(keep_alive: int | None) -> int:
        if keep_alive is None or keep_alive <= 0:
            return -1
        return int(keep_alive)

    def preload(self, *, keep_alive: int | None = None) -> None:
        """Load the query model into Ollama so the first search does not pay that cost."""
        if not self.config.enabled:
            return
        installed = self.list_models()
        name = self.config.query_model
        if not self.model_installed(name, installed):
            raise OllamaUnavailable(f"未安装 {name}")
        self._post(
            "/api/generate",
            {
                "model": name,
                "keep_alive": self._keep_alive_value(keep_alive),
            },
        )

    def unload(self) -> None:
        if not self.config.enabled:
            return
        for name in dict.fromkeys([self.config.query_model, self.config.vision_model]):
            if not name:
                continue
            try:
                self._post("/api/generate", {"model": name, "keep_alive": 0})
            except OllamaUnavailable:
                continue

    def expand_query(self, query: str, *, keep_alive: int | None = None) -> QueryPlan:
        base = static_expand(query)
        if not self.config.enabled:
            return base
        prompt = f"""你是插画图库的检索解析器。将用户条件改写成适合图文向量检索的描述，
尤其准确保留服装结构、袖型、领型、鞋袜、饰品、人体动作、身体朝向、镜头角度、
构图、色彩和画风。semantic_zh/semantic_en 必须忠实改写，不可添加用户未要求的主体。
positive_tags 与 negative_tags 使用简短英语视觉标签，空格改下划线；不要输出画师或角色名。
用户查询：{query}"""
        result = self._post(
            "/api/chat",
            {
                "model": self.config.query_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "keep_alive": self._keep_alive_value(keep_alive),
                "format": QUERY_SCHEMA,
                "options": {"temperature": 0},
            },
        )
        try:
            data = json.loads(result["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise OllamaUnavailable("Ollama 返回了无法解析的查询结构") from exc
        return merge_llm_plan(base, data)

    @staticmethod
    def _image_base64(path: Path) -> str:
        image = load_rgb(path)
        image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def caption_image(self, path: Path) -> dict[str, Any]:
        prompt = """请为本地插画检索生成严格、可见事实导向的结构化描述。重点辨认：
1. 每层服装的种类、廓形、领口、袖型、长度、材质与装饰；
2. 袜子、鞋、头饰、发型和小饰品；
3. 人物的姿势、关节弯曲、身体朝向、视线、手脚位置；
4. 景别、视角、透视、构图、配色、光线、画材感和情绪。
看不清的细节留空，不猜角色、画师、品牌或隐含故事。summary_zh 写一到三句中文。"""
        result = self._post(
            "/api/chat",
            {
                "model": self.config.vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [self._image_base64(path)],
                    }
                ],
                "stream": False,
                "think": False,
                "format": CAPTION_SCHEMA,
                "options": {"temperature": 0},
            },
        )
        try:
            return json.loads(result["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise OllamaUnavailable("Ollama 返回了无法解析的图片描述") from exc


def flatten_caption(caption: dict[str, Any] | None) -> str:
    if not caption:
        return ""
    values: list[str] = []
    for value in caption.values():
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value)
    return "，".join(filter(None, values))

