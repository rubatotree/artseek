from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable

from huggingface_hub import snapshot_download

from .config import AppConfig

StatusCallback = Callable[[str], None]


def pull_models(
    config: AppConfig,
    *,
    include_tagger: bool = True,
    include_llm: bool = False,
    status: StatusCallback | None = None,
) -> list[str]:
    downloaded: list[str] = []

    if status:
        status(f"下载图文向量模型：{config.embedding.model}")
    snapshot_download(repo_id=config.embedding.model)
    downloaded.append(config.embedding.model)

    if include_tagger and config.tagger.enabled:
        if status:
            status(f"下载插画标签模型：{config.tagger.model}")
        snapshot_download(repo_id=config.tagger.model)
        downloaded.append(config.tagger.model)

    if include_llm:
        executable = shutil.which("ollama")
        if not executable:
            raise RuntimeError("未找到 Ollama。请先安装 Ollama，再运行 --with-llm")
        for model in dict.fromkeys(
            [config.llm.query_model, config.llm.vision_model]
        ):
            if status:
                status(f"下载本地语言/视觉语言模型：{model}")
            subprocess.run([executable, "pull", model], check=True)
            downloaded.append(model)
    return downloaded

