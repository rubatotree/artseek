# 贡献指南

ArtSeek is a local-first illustration search tool. Please keep originals on disk (no copy, move, or upload), bind the web UI to localhost by default, and do not add tests that download large models or require a GPU.

ArtSeek 面向本地插画收藏检索。贡献时请保持这三条约束：原图不移动、不复制、不上传；网页服务默认只监听本机；测试不下载大模型、不依赖 GPU。

## 开发环境

需要 Python 3.11、Git 和 [uv](https://docs.astral.sh/uv/)。Windows 上请用仓库里的脚本换成 CUDA 版 PyTorch，否则索引会静默跑在 CPU 上。

克隆时带上子模块。Vditor 钉在 tag `v3.11.3`；上游不提交 `dist/`，本仓库也不提交构建产物。笔记编辑器需在子模块里构建一次：

```powershell
git clone --recurse-submodules <本仓库>
cd ArtSeek
git submodule update --init --recursive
Push-Location src/artseek/web/static/vendor/vditor
pnpm install
pnpm build
Pop-Location
uv venv --python 3.11
uv pip install --link-mode=copy --python .venv\Scripts\python.exe -e ".[all,vision,dev]"
.\scripts\install_cuda_torch.ps1
```

Linux：

```bash
git submodule update --init --recursive
(cd src/artseek/web/static/vendor/vditor && pnpm install && pnpm build)
uv venv --python 3.11
uv pip install --python .venv/bin/python -e '.[all,vision,dev]'
```

其中 `vision` 可选依赖用于 PixAI v0.9、视觉属性分类、头部焦点检测与 CCIP 角色特征所需的 ONNX Runtime（不安装时仍可运行核心测试与基础图文检索）。

## 检查

```bash
python -m pytest
ruff check src tests
```

架构与检索融合见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 提交

使用 [Conventional Commits](https://www.conventionalcommits.org/)：`feat(search): ...`、`fix(web): ...`。一次提交只包含一件完整的事。安全问题不要开公开 issue，见 [SECURITY.md](SECURITY.md)。
