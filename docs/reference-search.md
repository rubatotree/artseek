# 参考检索增强（Vision Attributes & CCIP）

本文概述 ArtSeek 为「找画材参考」设计的正交视觉特征通道：**轻量 ONNX 视觉属性提取** 与 **跨服装同角色（CCIP）检索**。该通道不参与主图文检索的倒数排名融合，避免破坏 `visual` + `descriptor` 的排序标定，而是作为独立的筛选维度、排序键与单图相似检索模式工作。

---

## 1. v2 已实现能力（轻量 ONNX 路线）

相比早期依赖重型第三方包装与独立 Qdrant 侧车的草案，v2 采用纯 `onnxruntime`（可选依赖组 `vision`）直接加载模型，并将结果统一写入 `artseek.sqlite3` 与 `ccip.f32.npy`：

### 1.1 视觉属性与头部焦点（`src/artseek/ml/vision_attributes.py`）

通过 `VisionAttributeAnalyzer` 与 `HeadFocalDetector` 提取结构化元信息（由 `v004_vision_attributes.py` 写入 `images` 表，并通过 `image_task_state` 追踪进度）：

- **头部检测与焦点（`deepghs/anime_head_detection`）**：计算图库缩略图裁切焦点（`focal_x`、`focal_y`）、景别分类 `shot_type`（`close_up` / `upper_body` / `full_body` / `group` / `scenery_or_object`），以及供 CCIP 使用的人物裁剪框 `character_crop_json`。
- **媒介类型（`deepghs/anime_classification`）**：区分 `illustration`、`comic`、`bangumi`、`3d`、`not_painting`，并写入 `media_type`。
- **分级门控（`deepghs/anime_rating`）**：预测 `rating`（`safe` / `r15` / `r18`），配合网页端显示门控与检索范围过滤。
- **黑白与完成度（`deepghs/monochrome_detect`、`deepghs/anime_completeness`）**：判断 `is_monochrome` 与 `polished` / `rough` 等属性标签。
- **AI 生成检测（`deepghs/anime_ai_check` + 内嵌元数据解析）**：结合 PNG/EXIF 中的 NovelAI / ComfyUI / SD WebUI 生成参数与模型分类结果写入 `is_ai` / `ai_source` / `ai_score`（对真实照片与 3D 渲染禁用 2D 动画模型误判）。
- **美学评分（`deepghs/anime_aesthetic`）**：计算加权 `aesthetic_score`，支持图库与纯筛选模式下按美学排序（`aesthetic_desc`）。

上述筛选条件统一汇入 `ImageScope`（`media_types`、`ratings`、`monochrome`、`exclude_ai`、`shot_types`），由 `ScopeResolver` 在 SQLite 层完成交集筛选。

### 1.2 跨服装同角色检索（`src/artseek/ml/ccip.py`）

- 使用 `deepghs/ccip_onnx`（`ccip-caformer-24-randaug-pruned/model_feat.onnx`），优先按 `character_crop_json` 裁剪人物上半身区域后放缩至 `384×384`，提取 768 维 L2 归一化特征存入 `{index_dir}/ccip.f32.npy`（与 `images.id` 行号对齐）。
- `VectorIndex` 管理 `visual`、`descriptor` 与可选 `ccip` 三路矩阵；`SearchService.search_by_character` 执行跨服装同角色近邻检索，并与当前 `ImageScope` 求交。
- 后台任务队列（`indexing/jobs.py`）提供 `vision_enrich`、`ccip` 与 `ccip_reset` 任务，按图片任务状态增量补全，不阻塞正常看图与搜图。

---

## 2. 后续规划（扩展方向）

在现有视觉属性与 `ccip.f32.npy` 基础设施之上，后续计划按需逐步演进以下能力：

1. **角色簇聚与二次分组**：以 `ccip.f32.npy` 近邻为候选，结合 CCIP MetricHead 复核与并查集构建角色相册（如自动聚合同角色 `初音未来` 的不同作品），在单角色视图内按标签差异拆分服装分支（例如 `常服 12 / 礼服 8 / 其余 17`），并在上线前完成本地库阈值标定。
2. **构图与差分折叠**：结合 `phash` 与视觉近邻识别同系列差分稿（卡片折叠与组内翻页，支持手动拆组覆盖）；补充 12 维构图布局向量（人物框中心/占比、头身比与留白）用于同构图检索。
3. **局部参考与配色/姿态**：按需提取主题主色板（CIE76 $\Delta E$ 复核）、人物关键点骨架叠加及手部/眼部局部参考裁剪。

---

## 3. 设计约束

- **按需启用**：视觉属性与 CCIP 模型均作为 `vision` 可选依赖管理，默认不阻塞基础 `SigLIP 2` + 标签索引；未执行对应后台任务时界面自动降级。
- **独立存储与可重建**：`ccip.f32.npy` 与 `vision_*` 列均为派生状态，可通过任务队列清理重算，绝不修改原始图片文件。
