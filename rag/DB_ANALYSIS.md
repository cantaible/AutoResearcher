# 📊 MariaDB 数据库分析报告

## 基本信息

| 项目 | 值 |
|------|-----|
| **数据库类型** | MariaDB (Docker 容器) |
| **容器名** | `news-reader-local-db` |
| **连接方式** | `127.0.0.1:3307` |
| **数据库名** | `news_reader` |
| **数据来源** | 新闻采集系统 (news harvester) 生产环境副本 |
| **只读模式** | ✅ 是 |
| **数据备份** | `news_reader_20260404_184849.sql.gz` (约 30MB) |

---

## 表结构总览

共 3 张业务表 + 3 张序列表：

### 1. `news_article` — 核心数据表 ⭐

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | BIGINT PK | 文章 ID |
| `title` | VARCHAR(255) | 标题 |
| `category` | ENUM(AI, GAMES, MUSIC, UNCATEGORIZED) | 分类 |
| `source_name` | VARCHAR(255) | 来源名称 |
| `sourceurl` | VARCHAR(1024) | 原始 URL |
| `published_at` | VARCHAR(255) | 发布时间 |
| `scraped_at` | VARCHAR(255) | 抓取时间 |
| `raw_content` | LONGTEXT | 文章正文（主要是 HTML）|
| `summary` | VARCHAR(255) | 摘要 |
| `tags` | VARCHAR(255) | 标签（当前全部为空）|
| `tumbnailurl` | VARCHAR(255) | 缩略图 URL |

### 2. `feed_item` — RSS 订阅源配置

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | BIGINT PK | 订阅源 ID |
| `name` | VARCHAR(255) | 名称 |
| `url` | VARCHAR(255) | RSS URL |
| `category` | ENUM | 分类 |
| `enabled` | BIT | 是否启用 |
| `source_type` | VARCHAR(255) | 来源类型 |
| `etag` / `last_modified` | VARCHAR | 缓存控制 |

### 3. `thumbnail_task` — 缩略图任务队列（空表）

---

## 数据统计

### 文章总量

| 指标 | 数量 |
|------|------|
| **总文章数** | **7,924** |
| 有正文 (raw_content) | 7,811 (98.6%) |
| 有摘要 (summary) | 7,607 (96.0%) |
| 有标题 (title) | 7,924 (100%) |
| 有来源 URL | 7,924 (100%) |
| 有标签 (tags) | 0 (0%) |

### 分类分布

| 分类 | 数量 | 占比 | 平均正文长度 |
|------|------|------|------------|
| **AI** | 4,732 | 59.7% | ~15,285 字符 |
| **GAMES** | 2,475 | 31.2% | ~711 字符 |
| **MUSIC** | 667 | 8.4% | ~4,903 字符 |
| UNCATEGORIZED | 50 | 0.6% | - |

### 内容格式

| 类型 | 数量 |
|------|------|
| **HTML 格式** | 7,805 (99.9%) |
| 纯文本 | 6 |

### 正文长度统计

| 指标 | 值 |
|------|-----|
| 平均长度 | ~9,801 字符 |
| 最大长度 | 161,801 字符 |
| 最小长度 | 12 字符 |

### 来源分布 (Top 10)

| 来源 | 数量 |
|------|------|
| AI hot | 2,000 |
| AI base | 1,327 |
| Pocket Gamer | 1,208 |
| eurogamer | 816 |
| GameIndustry.biz | 407 |
| Music Ally | 368 |
| 量子位 (QbitAI) | 329 |
| Music Business World | 299 |
| 新智元 (Aiera) | 244 |
| Search Engine Roundtable | 232 |

### 时间范围

| 指标 | 值 |
|------|-----|
| 最早文章 | 2024-08-14 |
| 最新文章 | 2026-04-04 |
| 跨度 | 约 20 个月 |

### 订阅源

- 共 **29 个** RSS 订阅源
- 涵盖中英文 AI/游戏/音乐领域

---

## RAG 转换关键考虑

> [!IMPORTANT]
> ### 需要用户决策的问题

1. **Embedding 模型选择**：
   - 本地模型（如 `sentence-transformers/all-MiniLM-L6-v2`，免费，英文优先）
   - 多语言模型（如 `BAAI/bge-m3`，支持中英文，更适合本数据集）
   - OpenAI API（如 `text-embedding-3-small`，付费但效果好）

2. **向量数据库选择**：
   - **ChromaDB** — 轻量级，纯 Python，适合本地开发
   - **FAISS** — Meta 出品，高性能，适合大规模
   - **Qdrant** — 功能丰富，支持过滤
   - **Milvus/Weaviate** — 更重量级

3. **文本处理策略**：
   - raw_content 是 HTML 格式（99.9%），需要 HTML → 纯文本清洗
   - AI 类文章平均 15K 字符，需要分块 (chunking)
   - 中英文混合内容需要考虑分词策略

4. **分块策略**：
   - 按固定字符数切分（简单但可能切断语义）
   - 按段落/标题语义切分（更智能）
   - 递归字符切分（LangChain 默认方式）
   - 建议 chunk size: 500-1000 tokens，overlap: 100-200 tokens

5. **元数据保留**：
   - `category`, `source_name`, `published_at` 可作为过滤条件
   - 支持按分类、来源、时间范围进行混合检索
