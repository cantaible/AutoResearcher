# Evidence Pool 实施 TODO

## 目标
在 State 中添加 evidence_pool 字段，结构化保存 RAG/Web Search 的原始证据，支持去重和评测。

---

## Phase 1: 基础设施（30 分钟）

### 1.1 添加去重工具函数
**文件**: `src/utils.py`

- [x] 添加 `generate_evidence_id(evidence: dict) -> str`
  - RAG: `f"rag_{article_id}"`
  - Web Search: `f"web_{url_hash}"`
  - 其他: `f"unknown_{content_hash}"`

- [x] 添加 `deduplicate_evidence(evidence_list: list[dict]) -> list[dict]`
  - 基于 `id` 字段去重
  - 保留最早添加的条目

### 1.2 修改 State 定义
**文件**: `src/state.py`

- [x] `RAGExecuteState` 添加: `evidence_pool: Annotated[list[dict], operator.add]`
- [x] `RAGResearcherState` 添加: `evidence_pool: Annotated[list[dict], operator.add]`
- [x] `ResearcherState` 添加: `evidence_pool: Annotated[list[dict], operator.add]`
- [x] `ResearcherOutputState` 添加: `evidence_pool: Annotated[list[dict], operator.add]`
- [x] `SupervisorState` 添加: `evidence_pool: Annotated[list[dict], operator.add]`
- [x] `AgentState` 添加: `evidence_pool: Annotated[list[dict], operator.add]`

---

## Phase 2: RAG 子图集成（45 分钟）

### 2.1 修改 rag_search 返回原始文章
**文件**: `rag/rag_search.py`

- [x] 在返回的 dict 中添加 `articles` 字段
  ```python
  "articles": [
      {
          "id": doc.metadata.get("id"),
          "title": doc.metadata.get("title"),
          "content": doc.page_content,
          "published_date": doc.metadata.get("published_date"),
          "url": doc.metadata.get("url"),
      }
      for doc in reranked_docs
  ]
  ```

### 2.2 修改 RAG execute 节点构建 evidence_pool
**文件**: `src/rag_subgraph.py`

- [x] 在 `execute()` 函数中，从 `rag_search` 结果提取 `articles`
- [x] 构建 evidence_items:
  ```python
  from utils import generate_evidence_id
  from datetime import datetime
  
  evidence_items = []
  for article in articles:
      evidence = {
          "source": "rag",
          "article_id": article.get("id"),
          "url": None,
          "title": article.get("title", ""),
          "content": article.get("content", ""),
          "published_date": article.get("published_date"),
          "used_by_node": "rag_researcher",
          "query": current_query,
          "timestamp": datetime.now().isoformat(),
      }
      evidence["id"] = generate_evidence_id(evidence)
      evidence_items.append(evidence)
  ```
- [x] 在返回的 dict 中添加: `"evidence_pool": evidence_items`

---

## Phase 3: Supervisor 汇总（20 分钟）

### 3.1 修改 supervisor_tools 汇总 evidence_pool
**文件**: `src/graph.py`

- [x] 在处理 `ConductResearch` 结果时，添加汇总逻辑:
  ```python
  from utils import deduplicate_evidence
  
  # 汇总 evidence_pool
  all_evidence = []
  for obs in tool_results:
      all_evidence.extend(obs.get("evidence_pool", []))
  
  # 去重
  all_evidence = deduplicate_evidence(all_evidence)
  
  if all_evidence:
      update_payload["evidence_pool"] = all_evidence
  ```

### 3.2 修改 ConductRAGResearch 处理
**文件**: `src/graph.py`

- [x] 在处理 `ConductRAGResearch` 结果时，同样汇总 evidence_pool

### 3.3 修改 research_supervisor 输出
**文件**: `src/graph.py`

- [x] 在 `research_supervisor` 子图的最终返回中添加:
  ```python
  "evidence_pool": state.get("evidence_pool", [])
  ```

---

## Phase 4: 持久化（10 分钟）

### 4.1 修改 runner.py 保存 evidence_pool
**文件**: `src/runner.py`

- [x] 在 `save_evidence_pool()` 中添加去重:
  ```python
  from utils import deduplicate_evidence
  
  def save_evidence_pool(run_dir: Path, evidence_pool: list):
      """保存证据池到 evidence_pool.json（去重后）。"""
      evidence_pool = deduplicate_evidence(evidence_pool)
      with open(run_dir / "evidence_pool.json", "w", encoding="utf-8") as f:
          json.dump(evidence_pool, f, ensure_ascii=False, indent=2)
  ```

---

## Phase 5: Web Search 集成（可选，30 分钟）

### 5.1 修改 Web Search 节点
**文件**: `src/graph.py` 或相关文件

- [ ] 找到 Web Search 工具调用的位置
- [ ] 构建 evidence_pool:
  ```python
  evidence_items = []
  for result in search_results:
      evidence = {
          "source": "web_search",
          "article_id": None,
          "url": result.get("url"),
          "title": result.get("title", ""),
          "content": result.get("snippet", ""),
          "published_date": result.get("date"),
          "used_by_node": state.get("current_node", "web_researcher"),
          "query": query,
          "timestamp": datetime.now().isoformat(),
      }
      evidence["id"] = generate_evidence_id(evidence)
      evidence_items.append(evidence)
  ```
- [ ] 在返回中添加: `"evidence_pool": evidence_items`

---

## Phase 6: 评测集成（30 分钟）

### 6.1 修改 eval_findings.py
**文件**: `eval/eval_findings.py`

- [x] 添加 `load_evidence_pool()` 函数
- [x] 修改 `compute_evidence_support()` 使用 evidence_pool:
  ```python
  evidence_pool = load_evidence_pool(run_dir)
  
  for match in matches:
      if not match.matched_event:
          continue
      
      # 从 evidence_pool 提取相关文章
      relevant_articles = [
          e["article_id"] for e in evidence_pool
          if e["source"] == "rag" 
          and e["article_id"] is not None
          and matches_event(e, match.matched_event)
      ]
      
      match.evidence_article_ids = relevant_articles
  ```

---

## Phase 7: 测试验证（30 分钟）

- [x] 运行完整研究流程
- [x] 检查生成的 `evidence_pool.json`:
  - [x] 结构是否正确
  - [x] article_id 是否正确
  - [x] 是否有重复条目（去重逻辑已测试通过）
- [ ] 运行评测脚本，验证 Evidence Support Rate
- [ ] 检查 evidence_pool 大小是否合理

---

## 预估总时间
- Phase 1-4（核心功能）: 约 1.5 小时
- Phase 5（Web Search）: 约 0.5 小时（可选）
- Phase 6-7（评测+测试）: 约 1 小时

**总计**: 2-3 小时

---

## 注意事项

1. **向后兼容**: 保留 raw_results 和 raw_notes，确保现有流程不受影响
2. **增量实施**: 先做 RAG（Phase 1-4），验证通过后再做 Web Search
3. **去重时机**: 在 supervisor_tools 汇总时去重（主要），在 runner.py 保存前再次去重（保险）
4. **ID 生成**: 确保 ID 唯一且稳定（同一篇文章总是生成相同 ID）
5. **性能影响**: evidence_pool 会增加 checkpoint 大小，但相比文本冗余仍然更优

---

## 完成标准

- [ ] 运行后生成 `evidence_pool.json`
- [ ] 文件包含所有 RAG 检索到的文章元数据
- [ ] 无重复条目
- [ ] 评测脚本能正确计算 Evidence Support Rate
- [ ] 所有测试通过
