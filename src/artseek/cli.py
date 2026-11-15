from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import torch
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from .compute import cuda_mismatch_hint, describe_runtime
from .config import AppConfig, default_config_path, load_config, write_default_config
from .image_io import canonical_path
from .indexer import ImageIndexer
from .library import index_roots_from_scope, load_catalog, load_scope, root_image_counts
from .llm import OllamaClient
from .model_manager import pull_models
from .provenance import (
    PIXIV_DOMAIN,
    TWITTER_DOMAIN,
    ProvenanceService,
    ProvenanceStore,
    RateLimiter,
    scan_payloads,
)
from .search import SearchEngine
from .server_output import server_output
from .store import VectorStore

app = typer.Typer(
    name="artseek",
    no_args_is_help=True,
    help="面向本地插画收藏的自然语言与以图搜图工具。",
)
console = Console()
provenance_app = typer.Typer(help="扫描与补全图片的来源元信息。")
app.add_typer(provenance_app, name="provenance")


def _config(path: Path | None) -> AppConfig:
    return load_config(path or default_config_path())


@app.command("init")
def init_command(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    force: Annotated[bool, typer.Option("--force", help="覆盖已有配置")] = False,
) -> None:
    """创建默认配置文件。"""
    path = write_default_config(config, force=force)
    console.print(f"配置文件：[bold]{path}[/bold]")


@app.command("pull-models")
def pull_models_command(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    with_llm: Annotated[
        bool, typer.Option("--with-llm", help="同时通过 Ollama 拉取查询扩写与看图模型")
    ] = False,
    no_tagger: Annotated[
        bool, typer.Option("--no-tagger", help="不下载插画专用标签器")
    ] = False,
) -> None:
    """预先下载模型；完成后检索阶段可以完全断网运行。"""
    cfg = _config(config)
    downloaded = pull_models(
        cfg,
        include_tagger=not no_tagger,
        include_llm=with_llm,
        status=lambda message: console.print(f"[cyan]→[/cyan] {message}"),
    )
    console.print(f"[green]完成：[/green]{len(downloaded)} 个模型已就绪。")


def _run_index(
    cfg: AppConfig,
    directories: list[Path],
    *,
    force: bool,
    captions: bool | None,
    recursive: bool,
) -> None:
    mismatch = cuda_mismatch_hint()
    if mismatch:
        console.print(f"[yellow]{mismatch}[/yellow]")
    console.print("正在载入本地向量模型与插画标签器……")
    indexer = ImageIndexer(cfg)
    console.print(f"推理设备：{describe_runtime(cfg.embedding.device)}")
    progress_state: dict[str, int] = {}
    try:
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            def update(done: int, total: int, root: str) -> None:
                if root not in progress_state:
                    progress_state[root] = progress.add_task(
                        f"索引 {Path(root).name or root}", total=max(total, 1)
                    )
                progress.update(progress_state[root], completed=done)

            summary = indexer.index(
                directories,
                recursive=recursive,
                force=force,
                captions=captions,
                progress=update,
            )
    finally:
        indexer.close()

    table = Table(title="索引完成")
    table.add_column("发现")
    table.add_column("新建/更新")
    table.add_column("未变化")
    table.add_column("移除失效")
    table.add_column("失败")
    table.add_row(
        str(summary.discovered),
        str(summary.indexed),
        str(summary.unchanged),
        str(summary.deleted),
        str(len(summary.failed)),
    )
    console.print(table)
    if summary.llm_warning:
        console.print(f"[yellow]图片长描述已回退：{summary.llm_warning}[/yellow]")
    for path, reason in summary.failed[:20]:
        console.print(f"[red]跳过[/red] {path}: {reason}")


@app.command("index")
def index_command(
    directories: Annotated[list[Path], typer.Argument(help="一个或多个图片目录")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    force: Annotated[bool, typer.Option("--force", help="重新计算已有图片")] = False,
    captions: Annotated[
        bool | None,
        typer.Option(
            "--captions/--no-captions",
            help="用本地视觉语言模型生成服装、动作与构图长描述",
        ),
    ] = None,
    recursive: Annotated[
        bool, typer.Option("--recursive/--no-recursive", help="递归扫描子目录")
    ] = True,
) -> None:
    """增量预处理一个或多个本地图片目录。"""
    _run_index(
        _config(config), directories, force=force, captions=captions, recursive=recursive
    )


@app.command("rebuild")
def rebuild_command(
    directories: Annotated[list[Path], typer.Argument(help="重建后要纳入的图片目录")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    captions: Annotated[bool, typer.Option("--captions")] = False,
) -> None:
    """清空旧向量索引后完整重建；不会删除或修改原图片。"""
    cfg = _config(config)
    store = VectorStore(cfg)
    try:
        store.clear()
    finally:
        store.close()
    _run_index(cfg, directories, force=True, captions=captions, recursive=True)


def _print_results(response: object) -> None:
    data = asdict(response)  # type: ignore[arg-type]
    plan = data["query_plan"]
    if plan["positive_tags"]:
        console.print("细节标签：" + ", ".join(plan["positive_tags"]))
    if plan["negative_tags"]:
        console.print("排除标签：" + ", ".join(plan["negative_tags"]))
    if data.get("llm_warning"):
        console.print(f"[yellow]已使用内置词典：{data['llm_warning']}[/yellow]")
    table = Table(show_lines=False)
    table.add_column("匹配", justify="right")
    table.add_column("尺寸")
    table.add_column("文件")
    table.add_column("命中细节")
    for result in data["results"]:
        table.add_row(
            f"{result['score']:.1f}",
            f"{result['width']}×{result['height']}",
            result["path"],
            ", ".join(result["tag_hits"]),
        )
    console.print(table)


@provenance_app.command("scan")
def provenance_scan_command(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    deep: Annotated[
        bool, typer.Option("--deep", help="打开所有图片读取嵌入元信息（较慢）")
    ] = False,
    only_new: Annotated[
        bool, typer.Option("--only-new", help="只处理尚未登记的点")
    ] = False,
) -> None:
    """只读扫描文件名和嵌入元信息，不访问网络。"""
    cfg = _config(config)
    store = ProvenanceStore(cfg.provenance_path)
    vectors = VectorStore(cfg)
    try:
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("扫描来源", total=max(vectors.count(), 1))
            summary = scan_payloads(
                store,
                vectors.iter_payloads(),
                total=vectors.count(),
                deep=deep,
                only_new=only_new,
                progress=lambda done, total, label: progress.update(
                    task,
                    completed=done,
                    total=max(total, 1),
                    description=f"扫描 {Path(label).name or ''}",
                ),
            )
    finally:
        vectors.close()
    table = Table(title="来源扫描完成")
    table.add_column("扫描")
    table.add_column("识别")
    table.add_column("未识别")
    table.add_column("新增")
    table.add_column("失效")
    table.add_row(
        str(summary["scanned"]),
        str(summary["identified"]),
        str(summary["unresolved"]),
        str(summary["new"]),
        str(summary["stale"]),
    )
    console.print(table)


@provenance_app.command("enrich")
def provenance_enrich_command(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    platform: Annotated[
        list[str] | None, typer.Option("--platform", help="pixiv 或 twitter，可重复")
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="只显示待补全数量")] = False,
) -> None:
    """按域名限速补全已识别的 Pixiv/X 来源。"""
    cfg = _config(config)
    store = ProvenanceStore(cfg.provenance_path)
    platforms = tuple(platform or ("pixiv", "twitter"))
    records = store.pending_records(platforms=platforms, limit=limit)
    if dry_run:
        console.print(f"待补全：{len(records)} 条")
        return
    if not records:
        console.print("没有待补全的来源。先运行 `artseek provenance scan`。")
        return
    service = ProvenanceService(
        store,
        pixiv_limiter=RateLimiter(
            store,
            PIXIV_DOMAIN,
            min_interval=max(0.0, cfg.provenance.pixiv_min_interval),
            daily_limit=max(1, cfg.provenance.pixiv_daily_limit),
        ),
        twitter_limiter=RateLimiter(
            store,
            TWITTER_DOMAIN,
            min_interval=max(0.0, cfg.provenance.twitter_min_interval),
            daily_limit=max(1, cfg.provenance.twitter_daily_limit),
        ),
    )
    try:
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("补全来源", total=len(records))
            summary = service.enrich_records(
                records,
                limit=limit,
                progress=lambda done, total, label: progress.update(
                    task, completed=done, total=max(total, 1), description=f"补全 {label or ''}"
                ),
            )
    finally:
        service.close()
    console.print(
        f"完成：成功 {summary['ok']}，失败 {summary['failed']}"
        + ("，已暂停" if summary.get("stopped") else "")
    )


@app.command("search")
def search_command(
    query: Annotated[str, typer.Argument(help="中文/日文/英文自然语言或关键词")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, max=500)] = 40,
    image: Annotated[
        list[Path] | None, typer.Option("--image", "-i", help="参考图，可重复传入多张")
    ] = None,
    text_weight: Annotated[
        float,
        typer.Option("--text-weight", min=0.0, max=8.0, help="文字在混合查询中的权重，默认 1"),
    
    ] = 1.0,
    no_llm: Annotated[
        bool, typer.Option("--no-llm", help="不调用 Ollama，仅用内置词典扩展")
    ] = False,
    strict_tags: Annotated[
        bool, typer.Option("--strict-tags", help="要求结果至少精确命中一个细节标签")
    ] = False,
    dedupe: Annotated[
        bool, typer.Option("--dedupe", help="相同文件或相同哈希的图片只保留最先检索到的")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """用自然语言、关键词，或参考图加文字组合查询。"""
    engine = SearchEngine(_config(config))
    try:
        response = engine.search(
            query,
            images=image or None,
            text_weight=text_weight,
            limit=limit,
            expand=not no_llm,
            strict_tags=strict_tags,
            dedupe=dedupe,
        )
    finally:
        engine.close()
    if json_output:
        console.print_json(json.dumps(asdict(response), ensure_ascii=False))
    else:
        _print_results(response)


@app.command("similar")
def similar_command(
    image: Annotated[Path, typer.Argument(help="参考图片")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, max=500)] = 40,
) -> None:
    """以图搜图。"""
    engine = SearchEngine(_config(config))
    try:
        response = engine.similar(image, limit=limit)
    finally:
        engine.close()
    _print_results(response)


@app.command("serve")
def serve_command(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 7860,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="模型就绪并开始监听后再打开浏览器")
    ] = False,
) -> None:
    """启动只监听本机的可视化图库界面。模型在首次检索时载入。"""
    cfg = _config(config)
    with server_output(cfg.data_dir):
        try:
            import uvicorn
        except ImportError as exc:
            raise typer.BadParameter('缺少网页依赖，请执行：pip install -e ".[web]"') from exc
        from .web.app import create_app

        mismatch = cuda_mismatch_hint()
        if mismatch:
            console.print(f"[yellow]{mismatch}[/yellow]")
        console.print("正在启动本地图库……")
        application = create_app(cfg)
        console.print(f"推理设备：{describe_runtime(cfg.embedding.device)}")
        console.print("图文向量模型将在首次检索时载入，闲置后自动卸载以释放显存。")
        console.print(f"[green]已就绪。[/green] 即将监听 http://{host}:{port}")
        if open_browser:
            import threading
            import webbrowser

            threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()
        uvicorn.run(application, host=host, port=port, log_level="info")


def _print_root_counts(rows: list[tuple[str, int]]) -> None:
    console.print(f"图片总数：[bold]{sum(count for _, count in rows)}[/bold]")
    if not rows:
        console.print("  （还没有索引任何目录）")
        return
    for root, count in rows:
        console.print(f"  {count:>7}  {root}")


@app.command("stats")
def stats_command(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """显示本地索引概况。优先读目录缓存，不打开向量库。"""
    cfg = _config(config)
    catalog = load_catalog(cfg.folder_catalog_path)
    if catalog is not None:
        _print_root_counts(root_image_counts(catalog))
        return
    console.print("[yellow]目录缓存缺失，正在扫描向量库（图片多时会较慢）…[/yellow]")
    try:
        store = VectorStore(cfg)
    except Exception as exc:
        console.print(f"[red]无法打开向量库：{exc}[/red]")
        roots = index_roots_from_scope(load_scope(cfg))
        if roots:
            console.print("根据界面选择缓存，索引根目录可能是：")
            for root in roots:
                console.print(f"         {root}")
        console.print(
            "若网页服务正在运行，等目录树加载完成后会生成 folder_catalog.json，"
            "之后 stats 可立即返回。"
        )
        return
    try:
        roots: dict[str, int] = {}
        for _, payload in store.iter_folder_payloads():
            root = str(payload.get("root") or payload.get("root_key") or "未知目录")
            roots[root] = roots.get(root, 0) + 1
        _print_root_counts(sorted(roots.items()))
    finally:
        store.close()


@app.command("doctor")
def doctor_command(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """检查 GPU、可选组件、Ollama 与路径配置。"""
    cfg = _config(config)
    config_path = config or default_config_path()
    cuda_available = torch.cuda.is_available()
    mismatch = cuda_mismatch_hint()
    if cuda_available:
        cuda_detail = f"{torch.cuda.get_device_name(0)} · {torch.__version__}"
    elif mismatch:
        cuda_detail = mismatch.replace("\n", " ")
    else:
        cuda_detail = "不可用"
    checks = [
        ("配置文件", str(config_path), config_path.exists()),
        ("数据目录", str(cfg.data_dir), cfg.data_dir.exists()),
        (
            "CUDA",
            cuda_detail,
            cuda_available,
        ),
        ("插画标签器", "timm", importlib.util.find_spec("timm") is not None),
        (
            "网页界面",
            "fastapi + uvicorn",
            importlib.util.find_spec("fastapi") is not None
            and importlib.util.find_spec("uvicorn") is not None,
        ),
        ("Ollama 命令", shutil.which("ollama") or "未找到", shutil.which("ollama") is not None),
        ("Ollama 服务", cfg.llm.base_url, OllamaClient(cfg.llm).is_available()),
    ]
    table = Table()
    table.add_column("项目")
    table.add_column("状态")
    table.add_column("详情")
    for name, detail, ok in checks:
        table.add_row(name, "[green]正常[/green]" if ok else "[yellow]未就绪[/yellow]", str(detail))
    console.print(table)
    console.print(f"路径规范化示例：{canonical_path(cfg.data_dir)}")


if __name__ == "__main__":
    app()
