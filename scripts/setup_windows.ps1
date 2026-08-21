$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "未找到 uv。请先执行：winget install --id Astral-sh.uv -e" -ForegroundColor Yellow
    exit 1
}

Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    uv venv --python 3.11
    # Windows 上 uv 默认 hardlink；venv 与 cache 跨盘时链接会 silently 失效，
    # 表现为 dist-info 还在、google.protobuf 源文件却丢失。
    uv pip install --link-mode=copy --python .venv\Scripts\python.exe -e ".[all]"
    # PyPI 上的 Windows torch 是 CPU 轮子，必须再换成 CUDA 构建，否则索引会一直走 CPU。
    & (Join-Path $PSScriptRoot "install_cuda_torch.ps1")
    .\.venv\Scripts\python.exe -m artseek.cli init
    Write-Host "安装完成。下一步：" -ForegroundColor Green
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host "  artseek pull-models --with-llm"
    Write-Host "  artseek index `"D:\Pictures\Illustrations`""
    Write-Host "  artseek serve"
}
finally {
    Pop-Location
}

