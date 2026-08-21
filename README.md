# ArtSeek

ArtSeek 是一个完全本地运行的插画图片收藏检索工具。它不会移动、复制或上传原图；首次建立索引后，可以用中文、日文、英文自然语言、细节关键词、参考图片，或“参考图 + 修改要求”查找图片。

它针对插画做了两层表示：

- `SigLIP 2` 多语言图文向量：负责主体、画风、氛围、整体构图和以图搜图；
- `WD EVA02` 插画标签 + 可选本地视觉语言模型描述：负责衣服层次、袖型、领口、鞋袜、饰品、动作、视角等细节。

检索时两路结果会融合。Ollama 可把“想找蓝白配色、飘逸披风、回眸构图，但不要夜景”拆成语义描述、正向细节和排除条件；Ollama 未运行时，程序会自动使用内置的中英日插画词典，不会导致基本检索失效。

## 功能

- 增量扫描一个或多个目录；只重算新增或发生变化的图片；
- 中文、日文、英文自然语言搜索；
- 服装、动作、镜头、构图、色彩等关键词搜索；
- `-关键词`、`不要……`、`排除……`等负向条件；
- 上传任意参考图进行以图搜图；
- 选中库内图片作为参考，再附加“保留服装但换成仰视”等文字条件；
- 可选严格标签模式，要求结果至少精确命中一个细节标签；
- 可选使用本地视觉语言模型，为每张插画生成结构化长描述；
- 命令行与离线本地网页图库；
- Windows、Linux；NVIDIA GPU、CPU 均可运行。

## 推荐配置

你的 RTX 4060 Laptop 8 GB 可以直接使用默认配置：

| 模块 | 默认模型 | 作用 |
|---|---|---|
| 图文向量 | `google/siglip2-base-patch16-256` | 多语言语义与以图搜图 |
| 插画标签 | `SmilingWolf/wd-eva02-large-tagger-v3` | 服装、动作、构图等 Danbooru 风格标签 |
| 查询理解 | `qwen3:4b`（Ollama，可选） | 把自然语言整理成检索计划 |
| 图片长描述 | `qwen3-vl:4b`（Ollama，可选） | 细拆服装层次、姿势和画面设计 |

默认图文向量模型约 4 亿参数，标签器约 3 亿参数。开启两个 Ollama 模型后，全部模型缓存通常需要数 GB 到十余 GB 磁盘空间，具体取决于 Ollama 的量化版本。

## Windows 快速安装

要求：Windows 10/11、NVIDIA 驱动、Python 3.11。推荐使用 [uv](https://docs.astral.sh/uv/) 管理环境。

```powershell
cd ArtSeek
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
```

如果已经安装 uv，也可以手动执行：

```powershell
uv venv --python 3.11
uv pip install --link-mode=copy --python .venv\Scripts\python.exe -e ".[all]"
.\scripts\install_cuda_torch.ps1
.\.venv\Scripts\Activate.ps1
```

Windows 上 PyPI 的 `torch` 默认是 CPU 轮子。`install_cuda_torch.ps1` 会把它换成 CUDA 13.0 构建；漏掉这一步时，索引会静默跑在 CPU 上，速度会慢一个数量级。可用 `artseek doctor` 确认是否显示 GPU 名称。

## Linux 安装

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -e '.[all]'
source .venv/bin/activate
```

## 第一次使用

### 1. 创建配置

```powershell
artseek init
artseek doctor
```

配置文件位于系统的用户配置目录。可以直接修改其中的模型、批大小、数据路径和检索权重。

### 2. 下载模型

基础功能只需 Hugging Face 上的图文模型和插画标签器：

```powershell
artseek pull-models
```

如需本地大模型理解查询和生成图片长描述，请先安装并启动 [Ollama](https://ollama.com/download)，然后运行：

```powershell
artseek pull-models --with-llm
```

下载完成后，索引与检索都可以断网运行。

### 3. 建立索引

```powershell
artseek index "D:\Pictures\Illustrations"
```

可以一次纳入多个目录：

```powershell
artseek index "D:\Pictures\References" "E:\Drawing\Collections"
```

以后收藏中加入新图，只需重复同一个 `index` 命令；未改变的图片会跳过。原图被删除后，相应索引项也会移除。

如果希望逐张生成更细的服装、动作和构图描述：

```powershell
artseek index "D:\Pictures\Illustrations" --captions --force
```

长描述会明显增加首次索引时间。建议先用默认的“向量 + 插画标签”索引验证检索效果，再决定是否在夜间为全部图片生成长描述。

### 4. 打开本地网页图库

```powershell
artseek serve
```

启动后立即监听 `http://127.0.0.1:7860`，图文向量模型默认不占显存；第一次检索时再载入，闲置 10 分钟无搜图后自动卸载。可在网页「高级」设置中手动加载/卸载，或改自动卸载时间。默认不打开浏览器。若要自动打开，使用 `artseek serve --open`。服务默认只监听本机。

## 查询示例

自然语言：

```powershell
artseek search "低饱和青蓝色调插画，少女穿宽松针织开衫站在窗边，柔和逆光，俯视构图"
artseek search "白色宽袖风衣配深色长裙，衣摆和发丝向画面外侧舒展，轻盈干净的剪影"
artseek search "雨夜霓虹与水面倒影，打透明雨伞的背影，冷暖对比光 -室内"
```

关键词和排除条件：

```powershell
artseek search "贝雷帽 风衣 围巾 飘雪 -室内" --strict-tags
artseek search "セーラー服 夕焼け 逆光 屋上"
```

以图搜图：

```powershell
artseek similar "D:\query.png" --limit 60
```

参考图加文字：

```powershell
artseek search "保持类似的服装轮廓，但改成仰视动态透视和蓝紫色" --image "D:\query.png"
```

机器可读 JSON：

```powershell
artseek search "泡泡袖 低头" --json
```

## 常用命令

| 命令 | 说明 |
|---|---|
| `artseek init` | 生成配置 |
| `artseek doctor` | 检查 CUDA、模型组件和 Ollama |
| `artseek pull-models` | 预下载图文向量与插画标签模型 |
| `artseek index DIR...` | 增量索引图片目录 |
| `artseek rebuild DIR...` | 清空向量库后重建；不碰原图 |
| `artseek search TEXT` | 文字或图文组合搜索 |
| `artseek similar IMAGE` | 以图搜图 |
| `artseek stats` | 查看各目录图片数量 |
| `artseek serve` | 启动本地网页界面；模型在首次检索时载入；加 `--open` 可打开浏览器 |

## 如何理解三种检索模式

1. **自然语言模式**：默认模式。整体语义与细节标签共同排序，适合“穿什么、怎么动、画面什么感觉”的描述。
2. **严格细节模式**：添加 `--strict-tags` 或勾选网页中的选项。结果必须至少命中一个解析出的插画标签，适合明确找某个衣服部件或动作。
3. **以图搜图/组合检索**：可添加多张参考图。文字与每张参考图的默认权重都是 1，点开参考图后可改该图权重；查询向量按权重加和后归一化。

网页中的分数只表示本次结果列表内的相对排序，不是概率。

## 数据与隐私

- 图片始终从本地路径读取；原图不会上传到远端；
- Qdrant 以嵌入式本地模式运行，无需 Docker 或数据库服务；
- Ollama 请求只发送到配置的 `127.0.0.1` 地址；
- 首次 `pull-models` 会联网下载模型；之后可以离线；
- 缩略图、模型缓存与向量数据库位于系统用户缓存/数据目录，不会写入图片收藏目录。

## 调优

### 显存不足

在配置中依次尝试：

```toml
[embedding]
batch_size = 8

[tagger]
batch_size = 2
```

仍不足时将 `device = "cpu"`，或临时关闭标签器：

```toml
[tagger]
enabled = false
```

更换向量模型后需要运行 `artseek rebuild ...`，因为不同模型的向量空间不可混用。

### 搜索太宽泛

- 使用具体部件：`泡泡袖、彼得潘领、水手领、玛丽珍鞋`；
- 使用具体动作：`回眸、撑伞、托腮、仰视`；
- 打开“细节标签至少命中一项”；
- 为图库运行一次 `--captions --force`；
- 在配置中把 `descriptor_weight` 从 `0.28` 提高到 `0.4` 左右。

### 搜索只看细节、不看画风

把 `visual_weight` 提高，同时降低 `descriptor_weight`。两者不要求严格相加为 1，程序会按当前检索通道重新归一化。

## 开发与测试

```bash
uv pip install --python .venv/bin/python -e '.[all,dev]'
python -m pytest
ruff check src tests
```

架构与检索融合细节见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 模型与许可证

- [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-256)：Apache-2.0；多语言图文检索；
- [WD EVA02 Large Tagger v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3)：Apache-2.0；插画多标签分类；
- [Qwen3](https://ollama.com/library/qwen3) 与 [Qwen3-VL](https://ollama.com/library/qwen3-vl)：通过 Ollama 可选安装；
- [Qdrant Python Client](https://github.com/qdrant/qdrant-client)：Apache-2.0，本项目使用其持久化本地模式。

ArtSeek 自身使用 MIT 许可证。模型权重仍受各自模型许可证约束。

