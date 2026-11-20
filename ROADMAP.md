# ArtSeek Roadmap

## v2（当前分支 `refactor/v2`）

v2 将运行时存储从 Qdrant + 多 sidecar 收敛为：

- 单个 WAL SQLite 元数据库 `{index_dir}/artseek.sqlite3`；
- 两个 `fp32` numpy 向量矩阵 `visual.f32.npy` / `descriptor.f32.npy`；
- 统一 `ImageScope` / `ScopeResolver` 与持久化后台任务队列。

设计与验收指标见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

| 里程碑 | 状态 | 验收结果 |
|---|---|---|
| M0 契约基线 | 完成 | v1 HTTP 契约测试 |
| M1 数据层 | 完成 | 数万张实图迁移计数一致；开库 183 ms，单通道检索 4.3 ms，tag 筛选 4.2 ms |
| M2 服务层 | 完成 | 服务层测试 + 覆盖索引；folder 列表 4.7 ms，scope 4.6 ms |
| M3 API | 完成 | 82 条 v1 路由兼容；冷启动 1.64 s；文本搜索 p50 92 ms；tag/作者筛选 <50 ms |
| M4 索引流水线 | 完成 | 1,000 实图 55.4 ms/张；数万图无变化全验 2.14 s / 增量 0.95 s；索引期间 Web p95 7.1 ms |
| M5 前端 | 待开始 | Vite + TypeScript + Svelte 5，替换单文件 `index.html`，逐项手测视图循环/localStorage/列数/滚动保留 |
| M6 收尾 | 待开始 | 删除 Qdrant 运行时与 v1 sidecar 读取代码；旧前端移除；README/ARCHITECTURE/CHANGELOG 收口 |

### 回滚

v1 数据文件从不被修改或删除。切回 `main`（或 tag `v1-baseline`）即可恢复 v1 服务。
