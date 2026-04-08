# RAG 搭建实施计划

## 目标

在 `rag/` 目录下搭建新闻检索 RAG 系统（MVP 版），从 MariaDB 读取 7,924 篇文章，embedding 到 ChromaDB，提供 `rag_search` tool 函数。

## 技术栈

| 组件 | 选型 |
|------|------|
| 向量数据库 | ChromaDB（持久化到 `rag/vectordb/`）|
| Embedding | BAAI/bge-m3（本地，MPS 加速）|
| HTML 清洗 | beautifulsoup4 + html2text |
| 数据源 | MariaDB `127.0.0.1:3307` |

## 产出文件

```
rag/
├── requirements.txt          # RAG 依赖
├── config.py                 # 连接配置（DB、ChromaDB 路径、模型名）
├── scripts/
│   ├── clean_html.py         # HTML → 纯文本清洗函数
│   └── build_index.py        # 一键脚本：读 MariaDB → embedding → 写 ChromaDB
├── rag_search.py             # rag_search tool 函数（对外接口）
├── vectordb/                 # ChromaDB 持久化数据（gitignore）
└── tests/
    └── test_rag_search.py    # 验证检索效果
```

## 实施步骤

### Step 1: 安装依赖

安装到 conda base：
```
chromadb, sentence-transformers, torch, beautifulsoup4, html2text, pymysql
```

---

### Step 2: config.py + clean_html.py

- `config.py`：MariaDB 连接信息、ChromaDB 路径、模型名、collection 名
- `clean_html.py`：`clean(html: str) -> str`，去标签，保留段落结构

---

### Step 3: build_index.py

核心脚本，做三件事：

1. 连接 MariaDB，读取所有 `news_article`（id, title, summary, category, source_name, published_at, raw_content）
2. 拼接 embedding 文本：`f"{title}\n{clean(summary) or ''}"`
3. 批量 embedding → 写入 ChromaDB，metadata 包含 `article_id`, `category`, `source_name`, `published_ts`

预期运行时间：2-4 分钟。

---

### Step 4: rag_search.py

```python
@tool
async def rag_search(
    query: str,
    time_range_days: int = 30,
    category: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """Search local news database."""
    # 1. 构造 ChromaDB where 条件（时间 + 分类）
    # 2. query embedding + metadata filter
    # 3. 用 article_id 回查 MariaDB 拿完整信息（标题、URL、摘要）
    # 4. 格式化返回
```

---

### Step 5: 测试验证

用 3 个查询验证效果：

1. `"最近一个月大模型新闻"` — 测时间过滤 + 语义
2. `"Coin Master 游戏"` — 测英文游戏类检索
3. `"OpenAI GPT"` — 测中英文混合语义匹配

验证指标：返回结果是否相关、时间是否正确、分类是否准确。

---

## 后续（MVP 跑通后）

- 增强版：新增 `news_chunks` collection，对 raw_content 分块 embedding
- 集成：将 `rag_search` 加入 deep research agent 的 `get_all_tools()`

## 开放问题

> [!IMPORTANT]
> **summary 清洗问题**：数据库中 `summary` 字段是 VARCHAR(255)，96% 有值。部分 summary 可能也含 HTML，需要和 raw_content 一样做清洗。`title` 都是纯文本无需清洗。

> [!IMPORTANT]
> **raw_content 在 MVP 中的角色**：MVP 阶段 raw_content 不参与 embedding，但 `rag_search` 返回结果时，是否需要回查 raw_content 并截取前 N 字符作为预览？还是只返回 title + summary + URL 就够了？
