$ErrorActionPreference = "Stop"

# Windows 上 PyPI 的 torch 默认是 CPU 轮子。NVIDIA 环境需要从 PyTorch 官方索引安装 CUDA 构建。
# 驱动 CUDA 13.1 可运行 cu130；若失败则回退到 cu126。

$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "未找到虚拟环境：$python" -ForegroundColor Red
    exit 1
}

$already = & $python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -eq 0) {
    & $python -c "import torch; print(f'已是 CUDA 构建：{torch.__version__} · {torch.cuda.get_device_name(0)}')"
    exit 0
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "未找到 uv。请先执行：winget install --id Astral-sh.uv -e" -ForegroundColor Yellow
    exit 1
}

$indexes = @(
    "https://download.pytorch.org/whl/cu130",
    "https://download.pytorch.org/whl/cu126"
)

uv pip uninstall --python $python torch torchvision
$installed = $false
foreach ($index in $indexes) {
    Write-Host "正在从 $index 安装 CUDA 版 PyTorch…" -ForegroundColor Cyan
    try {
        uv pip install --python $python torch torchvision --index-url $index
        if ($LASTEXITCODE -eq 0) {
            $installed = $true
            break
        }
    } catch {
        Write-Host $_.Exception.Message -ForegroundColor Yellow
    }
    Write-Host "从 $index 安装失败，尝试下一个索引。" -ForegroundColor Yellow
}

if (-not $installed) {
    Write-Host "CUDA 版 PyTorch 安装失败。" -ForegroundColor Red
    exit 1
}

& $python -c "import torch; assert torch.cuda.is_available(), '安装后 CUDA 仍不可用'; print(f'推理设备：{torch.cuda.get_device_name(0)} · {torch.__version__}')"
if ($LASTEXITCODE -ne 0) {
    exit 1
}
