# 插画来源元信息追溯

本文记录对现有收藏目录的实测结果、可自动追溯的规则、数据模型、限速与风控设计，以及实现状态。结论先行：

- 文件名规则可靠，可无网络解析出约 **66%** 的图源（Pixiv 作品 + X/Twitter 推文）。
- 剩余约 **34%** 是 Twitter CDN 短哈希名或自由命名，只能靠嵌入备注、画师目录或反向搜图。
- pixiv / X 的元信息抓取本身不难，真正的瓶颈是 **SauceNAO 的免费额度** 和 **pixiv 的反自动化策略**。
- 因此「一键补全所有图」必须做成 **有预算、可暂停、可续跑的任务队列**，而不是一个同步按钮。

> **实施状态（2026-11-16）**：解析器、`provenance.sqlite3`、Pixiv/FxTwitter 抓取、按域名限速与预算、查看器来源面板、作者筛选、索引后自动补全、CLI 与 API 均已实现。SauceNAO/ascii2d 按用户要求只做手动打开与 URL 粘贴，不配置 API key，也不使用 Pixiv 登录 Cookie。

---

## 1. 实测范围（已脱敏）

本文不记录本机收藏目录、作者账号或精确收藏数量。验证结论只保留文件名规则、嵌入来源字段、跨平台作者关联和解析优先级。


## 2. 解析规则

解析必须是纯函数、无网络、可单测。优先级：

1. **嵌入 URL**（最可信）：XMP `exif:UserComment`、EXIF `UserComment`、`XPComment`、`ImageDescription`、PNG `Comment`/`Description`、XMP `dc:description`。
2. **文件名**。
3. **目录名**：`画师/<作者名>/...` 只作为 `author_hint`，置信度低，不自动覆盖网络结果。
4. **未解析**：进入反向搜图队列或人工粘贴。

建议的正则：

```python
PIXIV = re.compile(
    r"^(?P<id>\d{5,10})"
    r"(?:[-_][0-9a-fA-F]{8,})?"       # 下载哈希
    r"_p(?P<page>\d+)"
    r"(?:[ _](?:master\d+|square\d+|.*))?$"  # _master1200 / 标题
)
TWITTER = re.compile(
    r"^(?P<id>\d{15,19})_(?P<index>\d+)(?:[ _].*)?$"
)
PIXIV_NAMED = re.compile(r"pixiv[\s_-]*id[\s_-]*(?P<id>\d+)", re.I)
```

解析前先做一次「展示名清洗」：

```text
去掉末尾的 " (1)"、" (2)" ...
去掉扩展名
保留原始文件名，解析结果不修改磁盘文件
```

URL 归一化：

| 输入 | 归一化 |
|---|---|
| `https://www.pixiv.net/artworks/123` | `pixiv:123` |
| `https://www.pixiv.net/en/artworks/123` | `pixiv:123` |
| `https://www.pixiv.net/i/123` | `pixiv:123` |
| `https://www.pixiv.net/member_illust.php?mode=medium&illust_id=123` | `pixiv:123` |
| `https://pixiv.me/foo` | 跟随一次重定向后再解析 |
| `https://x.com/user/status/123/photo/2` | `twitter:123`, photo=2 |
| `https://twitter.com/user/status/123` | `twitter:123`, photo=unknown |
| `https://vxtwitter.com/...` / `fxtwitter.com/...` | 按 `x.com` 处理 |
| `https://mobile.twitter.com/...` | 按 `x.com` 处理 |
| 其他 `http(s)://...` | `other`，原样保存，不自动补全 |

X 的嵌入备注通常只写 `.../status/{id}`，不写 `/photo/N`。实测样本中的备注 URL 普遍没有 photo 序号，因此：

- 有 `/photo/N` 时直接用 N（1 起）；
- 只有 status ID 时，`photo` 记为未知；抓取后用本地宽高在 `media.all` 中匹配；
- 若多条媒体宽高相同、无法唯一匹配，标记 `ambiguous_page`，保留状态级元信息，不猜序号；
- 文件名里的 `_{index}` 是可靠序号，优先于备注。

冲突处理：

- 文件名与嵌入 URL 指向不同作品时，两条都保留，展示 `conflict`，以嵌入 URL 为默认，等待用户确认。
- 同一 `content_hash` 的多份拷贝共享同一条逻辑来源（通过 `content_hash` 归并），存储上仍每个 `point_id` 一行；不同文件名解析出不同来源时，说明至少一个文件是错配或改图，标记冲突。
- 解析结果只写 sidecar，**不改原图、不改文件名、不写回图片元数据**。

---

## 3. 数据模型

### 3.1 为什么用 SQLite sidecar

来源元信息是「用户状态」，不是索引状态。它必须满足：

- 重新索引、换向量模型、移动文件后仍然存在；
- 可以手动编辑并覆盖自动结果；
- 可以记录抓取状态、失败原因、重试时间；
- 可以按作者过滤，而不必把全部 payload 重写一遍。

因此放在 `{index_dir}/provenance.sqlite3`，与 `collections.json`、`index_state.sqlite3` 同级，随索引 sidecar 一起迁移/备份。WAL 模式，单写锁。

### 3.2 表

```sql
CREATE TABLE image_source (
    point_id        TEXT PRIMARY KEY,
    content_hash    TEXT,
    path_key        TEXT,
    platform        TEXT NOT NULL,      -- pixiv | twitter | other
    source_id       TEXT NOT NULL,
    page            INTEGER,            -- Pixiv 0 起；X 1 起；NULL = 未知
    url             TEXT NOT NULL,
    candidates_json TEXT,               -- 文件名/嵌入 URL 冲突时的候选
    status          TEXT NOT NULL,      -- pending | ok | not_found | protected |
                                        -- rate_limited | error | unresolved | stale
    confidence      REAL NOT NULL,
    title           TEXT,
    description     TEXT,
    author_key      TEXT,
    posted_at       TEXT,
    width           INTEGER,
    height          INTEGER,
    page_count      INTEGER,
    tags_json       TEXT,
    raw_json        TEXT,
    source_origin   TEXT NOT NULL,      -- filename | embedded | manual
    manual_fields   TEXT,               -- 手动覆盖的字段名数组
    fetched_at      INTEGER,
    error           TEXT,
    updated_at      INTEGER NOT NULL
);

CREATE INDEX idx_source_hash ON image_source(content_hash);
CREATE INDEX idx_source_author ON image_source(author_key);
CREATE INDEX idx_source_platform_status ON image_source(platform, status);

CREATE TABLE author (
    key             TEXT PRIMARY KEY,  -- pixiv:1554775 / twitter:yoneyamai
    platform        TEXT NOT NULL,
    platform_id     TEXT NOT NULL,
    name            TEXT,
    handle          TEXT,
    profile_url     TEXT,
    avatar_url      TEXT,
    social_json     TEXT,
    profile_fetched INTEGER NOT NULL DEFAULT 0,
    updated_at      INTEGER NOT NULL,
    UNIQUE(platform, platform_id)
);

CREATE TABLE author_alias (
    alias_key     TEXT PRIMARY KEY,
    canonical_key TEXT NOT NULL,
    evidence      TEXT,                 -- pixiv_social / twitter_bio / manual
    confidence    REAL NOT NULL,
    updated_at    INTEGER NOT NULL
);

CREATE TABLE fetch_state (
    domain         TEXT PRIMARY KEY,    -- pixiv.net / api.fxtwitter.com
    blocked_until  INTEGER,
    last_request   REAL,
    day            TEXT,
    day_count      INTEGER,
    long_remaining INTEGER,
    last_error     TEXT
);
```

`point_id` 是主键，因为大多数元信息是「这张图」的属性；`content_hash` 用于文件移动/重命名后的归并。旧 payload 没有 `content_hash` 时，在首次写入来源记录时补算。

来源标签（Pixiv tag、推文 hashtag）只写入 `tags_json` 并展示，**不写入现有 `tags` payload**。现有 `tags` 是 WD 标签器的检索字段，混入日文/中文来源标签会污染关键词重排与标签命中。将来若需要按来源标签筛选，再加独立的 `source_tags` 投影。

### 3.3 索引生命周期

每次索引完成后做一次 `reconcile`（只读 SQLite + 本轮新增/变更点）：

- 新点：解析文件名/嵌入元信息，插入 `pending` 记录。
- 同一路径、`content_hash` 变化：说明文件被替换，旧记录标记 `stale`，不继承到新图。
- 同一 `content_hash`、`point_id` 变化：文件被移动/重命名，把记录的 `point_id` 更新到新点。
- 文件消失：记录保留，标记 `point_missing`，不删除用户手动填写的元信息。
- 多份拷贝共享同一 `content_hash`：各自有 `point_id` 行；手动编辑时在同一事务里同步所有同哈希行，避免改一份漏一份。

### 3.4 作者组

- 每个平台账号先建一个 `author`。
- Pixiv 作者资料里的 `social.twitter.url` → `author_alias`，置信度高。
- Twitter 作者简介/主页/推文中的 pixiv URL → `author_alias`，置信度中。
- `画师/<name>/` 目录 → `author_alias`，置信度低，默认只做提示，不自动合并。
- 一个作者组可以有多个平台账号（例如 Pixiv + Twitter + 小号）。按作者筛选时取该组所有 key 的并集。
- 不自动把「同名」当成同一人；同名只作为候选，需手动确认。

---

## 4. 抓取器与限速

### 4.1 端点与实测

| 平台 | 端点 | 实测/文档 | 建议 |
|---|---|---|---|
| Pixiv 作品 | `https://www.pixiv.net/ajax/illust/{id}` | 未登录可返回公开作品；限流、Cloudflare、reCAPTCHA | 1 请求 / 2 秒，单并发 |
| Pixiv 分页 | `https://www.pixiv.net/ajax/illust/{id}/pages` | 返回每页原图 URL、宽高 | 与作品信息合并，最多 2 请求/作品 |
| Pixiv 作者 | `https://www.pixiv.net/ajax/user/{uid}?full=1` | 返回 `social.twitter.url` | 每个作者只取一次 |
| X/Twitter | `https://api.fxtwitter.com/status/{id}` | 公开 API；文档标称 1000 请求/分钟/IP | 1–2 请求/秒，单并发 |
| SauceNAO | `https://saucenao.com/search.php?output_type=2&api_key=...` | 响应头含 `short_remaining`（30 秒）与 `long_remaining`（24 小时）；403 = key 无效；非 200 = 过载或额度耗尽 | 4 请求 / 30 秒；每日额度以 `long_remaining` 为准 |
| ascii2d | `https://ascii2d.net/search/file` | 无公开 API；Cloudflare 会拦截脚本；HTML 结构易变 | 仅手动触发，1 请求 / 10 秒，失败即停 |

### 4.2 风险矩阵

| 风险 | 触发条件 | 后果 | 缓解 |
|---|---|---|---|
| Pixiv IP 限流 | 短时间大量 `ajax/illust` | 429 / 403 / Cloudflare 挑战 | 2 秒间隔 + 抖动；收到 429/403 立即停 30 分钟，指数退避到 24 小时 |
| Pixiv 账号风险 | 使用自己的 `PHPSESSID` 批量请求 | 账号被限流或封禁 | **默认不带 Cookie**；高级设置中明确警告，仍建议不要用 |
| Pixiv ToS | 「通常范围を超えてサーバーに負担」/「機械的に大量操作」 | 法律/账号风险 | 只做个人元数据、低频率、可暂停、不绕过 CAPTCHA、不转售数据 |
| X 直接抓取 | 请求 `x.com` 页面/API | 需要登录、ToS 禁止、容易封 IP | 不直接抓 X；只调用 FxTwitter 的公开 JSON |
| FxTwitter 过载 | 高频请求 | 429 / 5xx | 1–2 请求/秒；本地缓存 30 天；失败退避 |
| SauceNAO 额度 | 免费 key 约 100 次/天 | 当天无法继续 | 队列 + 每日预算；`long_remaining` 到 0 立即停；优先处理未解析的图 |
| ascii2d 反爬 | 脚本请求 | 403 / 验证页 | 只做单张手动查询；失败后引导用户在浏览器打开；不自动重试 |
| 反向搜图隐私 | 上传图片到第三方 | 图片离开本机 | 只上传缩略图（≤512px），UI 明确提示；默认关闭自动上传 |
| 误判 | 文件名被改过 / 多图顺序变化 | 元信息挂错图 | 尺寸校验；置信度；冲突可见；手动覆盖优先 |

### 4.3 调度器

所有外网请求走一个单例 `FetchScheduler`：

```text
per-domain 单并发
per-domain 最小间隔（pixiv 2s, fxtwitter 1s, saucenao 7.5s, ascii2d 10s）
全局指数退避：30m → 2h → 12h → 24h
收到 403/429：立即写入 blocked_until，当前任务暂停，不重试
每日预算：pixiv 2,000；fxtwitter 5,000；saucenao = long_remaining；ascii2d 200
```

预算与 `blocked_until` 持久化在 `fetch_state`。进程重启后继续遵守，不会「一启动就冲一波」。

### 4.4 缓存策略

- 成功结果缓存 30 天；作品标题/作者改名不频繁，30 天足够。
- 404 / protected 缓存 7 天；不要每次打开图都重新请求已删除作品。
- 429 / 403 不缓存结果，只记 `blocked_until`。
- 打开查看器 **不触发网络请求**；只读本地 sidecar。只有用户点「刷新」或批量任务才发请求。

---

## 5. 批量补全与增量

### 5.1 三档动作

1. **本地扫描（秒级）**：只读文件名/嵌入元信息，写 `image_source`，不联网。对所有图都可执行。
2. **补全已识别来源（小时级）**：只请求已有 `pixiv:{id}` / `twitter:{id}` 的记录。默认动作。
3. **反向搜图（天/周级）**：SauceNAO / ascii2d，只处理未识别的图；必须手动确认预算。

估算（单并发）：

- <pixiv-work-count> 个 Pixiv 作品 × 2 请求 / 2 秒 ≈ 4.5 小时。
- <twitter-status-count> 条推文 / 1 请求/秒 ≈ 4.8 小时。
- <unresolved-count> 张未识别图，SauceNAO 100 次/天 ≈ **150 天**；这是必须提前告诉用户的数字。
- ascii2d 不稳定，不作为批量方案。

### 5.2 新图自动补全

在现有 `run_index_job` 成功后追加一个轻量步骤：

1. 只对本轮新增/变更的点做本地解析；
2. 若 `auto_enrich` 开启，把 `pixiv`/`twitter` 记录放入队列；
3. 后台 worker 按预算慢慢跑，索引任务本身不等待网络；
4. 打开网页时显示「来源补全中 123/<twitter-status-count>」；
5. 永不自动调用 SauceNAO/ascii2d。

### 5.3 手动粘贴

用户粘贴 URL 后：

1. 立即归一化成 `pixiv:{id}` / `twitter:{id}`；
2. 若已有缓存，直接返回；
3. 否则入队并返回 `pending`，UI 轮询或稍后刷新；
4. 也允许直接粘贴作者主页 URL，建立 `author_alias`；
5. 手动编辑的字段标记 `manual_override`，自动刷新不覆盖，除非用户点「恢复自动」。

---

## 6. API 与 UI

### 6.1 API

```text
GET  /api/images/{point_id}/provenance
PUT  /api/images/{point_id}/provenance          # 手动编辑
POST /api/images/{point_id}/provenance/resolve  # 粘贴 URL / 重新解析
POST /api/provenance/scan                       # 本地扫描
POST /api/provenance/enrich                     # 入队/启动任务
GET  /api/provenance/job                        # 进度、预算、blocked_until
GET  /api/provenance/stats                      # 统计 + 任务状态
POST /api/provenance/job/pause
POST /api/provenance/job/resume
POST /api/provenance/job/stop
GET  /api/authors?q=...                         # 作者搜索
GET  /api/authors/{key}                         # 作者组详情
GET  /api/authors/{key}/images                  # 作者组下的图（分页）
```

SPA 路由增加 `/author/{key}`，直接打开作者页时也能看到该作者的全部图。

现有搜索/目录/收藏/随机接口增加可选 `author_key` 参数。实现上复用现有的 `point_ids` 子集路径：sidecar 返回该作者组的 point ID 列表，再与目录/收藏范围求交，交给 `VectorStore` 的 subset/postfilter 逻辑。

搜索框支持过滤器语法：

```text
author:pixiv:1554775
author:米山舞
author:@yoneyamai
tag:オリジナル tag:女の子
```

`author:` / `artist:` 会解析为作者组（完整 key、显示名或 handle）；`tag:` 匹配已保存的 Pixiv 标签/推文 hashtag，多个 tag 取交集。作者与标签同时存在时取交集。只有过滤器、没有自由文本时走 payload 分页，不做向量检索。

### 6.2 查看器

「来源扫描与补全」位于左侧「高级」面板，不再占用搜索栏。图片查看器「笔记」旁增加「来源」按钮，打开右侧信息栏：

- 来源徽标（Pixiv / X / 其他）、标题/推文正文、作者（可点击）、发布时间、页数/图数；
- 作者旁显示 P / X / ↗ 平台图标，分别跳转到对应平台主页；
- 标签 chips；
- 文件路径（可复制）；
- 状态：`已补全 / 已删除 / 受保护 / 限流中 / 待补全 / 未识别`；
- URL 输入框 + 「解析 URL 并补全」；Pixiv/X 会自动抓取，其他 `http(s)` 来源原样保存；
- 「重新抓取」「打开原帖」「SauceNAO」「ascii2d」「复制原图路径」；
- 「保存来源」「恢复自动」。

查看器底部标题显示为「作者名 · 标题/推文正文 · 标签」：作者名可点击进入作者页或在当前范围筛选，标题/正文可点击打开原帖；没有标题/正文时回退到文件路径。

### 6.3 图库

- 查看器里的作者名点击后，当前范围（搜索结果/目录/收藏/随机）追加 `author` 过滤 chip；
- 没有图库范围（直接打开 `/i/{id}`）时，自动进入 `/author/{key}` 作者页；
- chip 可移除，URL 持久化；
- 批量菜单位于左侧「高级」面板，居中弹出，显示各平台文件数、作者组、今日额度；
- 任务面板显示暂停/继续/停止、当前域名、`blocked_until`、今日已用。

---

## 7. 失败模式与数据质量

| 场景 | 处理 |
|---|---|
| 作品/推文已删除 | `not_found`，保留 ID/URL，7 天后可重试；UI 显示「原帖已不可用」 |
| 受保护/仅登录 | `protected`，不尝试绕过；提示手动确认 |
| 文件名 ID 错 | 尺寸不匹配 → `confidence` 降低，标记需人工确认；不覆盖已有手动记录 |
| 多图顺序变化 | 用 `media.all` 的顺序；若尺寸不匹配，降级为「同推文，但页码不确定」 |
| 转推/引用 | 作者取原推作者；额外记录 `reposted_by` / `quoted`，筛选时默认按原推作者 |
| 同一图多份拷贝 | 以 `content_hash` 共享记录；不同文件名冲突时展示冲突列表 |
| 作者改名/换号 | 平台 ID 是主键，显示名可更新；别名表保留旧 handle |
| 图片被重新压缩 | 尺寸可能变化，不能仅凭尺寸否定；仍保留 ID 结果，置信度略降 |
| 网络失败 | `error` + 退避；不阻塞其他图 |

---

## 8. 分阶段实现

### Phase 0：解析器 + 离线报告（无网络）— 已完成

- `provenance.parser`：文件名/URL/XMP 解析，纯函数。
- CLI：`artseek provenance scan [--deep] [--only-new]` 输出识别/未识别/新增/失效统计。
- 单测覆盖：Pixiv 多页、哈希后缀、`_master1200`、`(1)`、18/19 位推文、XMP URL、冲突。
- 验收：对当前 <library-size> 张图复现本文统计。

### Phase 1：sidecar + API + 手动粘贴 — 已完成

- SQLite schema、迁移、WAL、备份/迁移接入。
- `GET/PUT /api/images/{point_id}/provenance`。
- URL 归一化与手动编辑。
- 验收：手动粘贴一条 pixiv/X URL，刷新后仍在；重启后仍在。

### Phase 2：Pixiv / FxTwitter 抓取 + 调度器 — 已完成

- `FetchScheduler`、预算、退避、缓存。
- Pixiv 作品/分页/作者；FxTwitter 推文/作者。
- 作者组与 `author_alias`。
- 验收：限速任务跑 50 条，记录耗时/429/404；断点续跑。

### Phase 3：查看器 + 作者筛选 — 已完成

- 来源面板、URL 编辑、刷新。
- `author_key` 过滤接入搜索/目录/收藏/随机。
- 验收：同一作者的 Pixiv + X 图能被一个 chip 筛出。

### Phase 4：反向搜图 — 已完成（手动模式）

- 按用户要求不配置 SauceNAO API key；查看器提供 SauceNAO / ascii2d 打开按钮和 URL 粘贴框。
- 不自动上传图片；用户自行在浏览器中查找。
- 粘贴作品/推文 URL 后立即解析并补全元信息。

### Phase 5：索引后自动补全 + 文档 — 已完成

- `run_index_job` 完成后启动 `only_new` 本地扫描；`auto_enrich=true` 时自动进入低速网络队列。
- 支持 `artseek provenance scan|enrich`、侧车迁移、WAL checkpoint。
- 验收：新放入一张 `{pid}_p0.jpg`，索引完成后自动出现来源记录。

---

## 9. 已确定的策略

1. **SauceNAO**：不配置 API key，只手动打开并粘贴 URL。
2. **Pixiv Cookie**：不使用登录 Cookie；公开作品直接读取元信息。
3. **作者筛选**：有图库范围时在当前范围内过滤；直接打开图片时进入 `/author/{key}` 作者页。
4. **自动补全**：`[provenance] auto_enrich = true` 默认开启，只自动补 Pixiv/X；不自动反向搜图。
5. **限速**：Pixiv 2 秒/请求、2000/天；FxTwitter 1 秒/请求、5000/天；可在 `config.toml` 调整。

---

## 10. 参考

- Pixiv 反爬与限流说明：<https://inside.pixiv.blog/2023/05/17/102629>
- Pixiv 服务利用规约（第 14 条禁止行为）：<https://policies.pixiv.net/>
- Pixiv robots.txt：<https://www.pixiv.net/robots.txt>
- FxTwitter / FxEmbed API 文档（1000 请求/分钟/IP）：<https://docs.fxembed.com/api/introduction/>
- SauceNAO API 示例（`short_remaining` / `long_remaining`、退避建议）：<https://saucenao.com/tools/examples/api/identify_images_v1.1.py>
- ascii2d robots.txt（`Content-Signal: ai-train=no`）：<https://ascii2d.net/robots.txt>
