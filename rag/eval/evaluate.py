"""RAG 分块策略对比实验。
5 种策略 × 7 条查询，用 OpenAI API 做 embedding + LLM-as-Judge 评分。
"""
import json, time, re, os, sys, concurrent.futures
from pathlib import Path
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

DATA_PATH = Path(__file__).parent / "results" / "subset_100.json"
RESULTS_DIR = Path(__file__).parent / "results"
client_ai = OpenAI()  # 从环境变量读 OPENAI_API_KEY

# ============ 分块策略定义 ============

def no_chunk(articles):
    """策略A: 不分块，只用标题+摘要"""
    docs, metas = [], []
    for a in articles:
        docs.append(f"{a['title']}\n{a['summary']}")
        metas.append({"article_id": a["id"], "source": a["source_name"]})
    return docs, metas

def recursive_char(articles, size=1000, overlap=200):
    """策略B: 递归字符切分（当前方案）"""
    sp = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    docs, metas = [], []
    for a in articles:
        for chunk in sp.split_text(a["raw_content"]):
            docs.append(chunk)
            metas.append({"article_id": a["id"], "source": a["source_name"]})
    return docs, metas

def sentence_based(articles, max_size=1000):
    """策略C: 按句子切，句子累积到 max_size 就开新块"""
    docs, metas = [], []
    for a in articles:
        sentences = re.split(r'(?<=[。！？.!?\n])', a["raw_content"])
        buf = ""
        for s in sentences:
            if len(buf) + len(s) > max_size and buf:
                docs.append(buf.strip())
                metas.append({"article_id": a["id"], "source": a["source_name"]})
                buf = ""
            buf += s
        if buf.strip():
            docs.append(buf.strip())
            metas.append({"article_id": a["id"], "source": a["source_name"]})
    return docs, metas

def semantic_chunk(articles, threshold=0.3):
    """策略D: 语义分块——相邻句子 embedding 相似度低于阈值时切"""
    docs, metas = [], []
    for a in articles:
        sents = [s for s in re.split(r'(?<=[。！？.!?\n])', a["raw_content"]) if s.strip()]
        if len(sents) <= 2:
            docs.append(a["raw_content"])
            metas.append({"article_id": a["id"], "source": a["source_name"]})
            continue
        # 批量获取句子 embedding
        embs = get_embeddings([s for s in sents])
        buf, buf_idx = [sents[0]], 0
        for i in range(1, len(sents)):
            sim = cosine_sim(embs[i-1], embs[i])
            if sim < threshold and len("".join(buf)) > 200:
                docs.append("".join(buf))
                metas.append({"article_id": a["id"], "source": a["source_name"]})
                buf = []
            buf.append(sents[i])
        if buf:
            docs.append("".join(buf))
            metas.append({"article_id": a["id"], "source": a["source_name"]})
    return docs, metas

def parent_child(articles, parent_size=2000, child_size=500):
    """策略E: 父子块——小块检索，匹配后返回大块上下文"""
    sp_parent = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=0)
    sp_child = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=100)
    docs, metas = [], []
    for a in articles:
        parents = sp_parent.split_text(a["raw_content"])
        for pi, parent in enumerate(parents):
            children = sp_child.split_text(parent)
            for child in children:
                docs.append(child)  # 用小块做检索
                metas.append({"article_id": a["id"], "source": a["source_name"],
                              "parent_text": parent[:800]})  # 存大块供返回
    return docs, metas

# ============ 工具函数 ============

def get_embeddings(texts, model="openai/text-embedding-3-small"):
    resp = client_ai.embeddings.create(input=texts, model=model)
    return [d.embedding for d in resp.data]

def cosine_sim(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = sum(x*x for x in a)**0.5
    nb = sum(x*x for x in b)**0.5
    return dot / (na * nb) if na and nb else 0

QUERIES = [
    "最近大模型有什么进展",
    "OpenAI GPT-5",
    "图像生成模型 2026年3月",
    "AI Agent 编程助手",
    "大模型安全与对齐",
    "Stable Diffusion",
    "国产大模型 开源",
]

def judge_relevance(query, doc_text):
    """LLM 判断检索结果和查询的相关性，返回 0/1/2 分"""
    resp = client_ai.chat.completions.create(
        model="openai/gpt-4.1-mini",
        messages=[{"role":"user", "content": f"""判断以下检索结果与查询的相关性。

查询: {query}
检索结果: {doc_text[:500]}

只回答一个数字:
2 = 高度相关（直接回答查询）
1 = 部分相关（相关领域但不直接回答）
0 = 不相关"""}],
        temperature=0, max_tokens=20)
    try: return int(resp.choices[0].message.content.strip()[0])
    except: return 0

# ============ 实验主流程 ============

def run_experiment():
    articles = json.loads(DATA_PATH.read_text("utf-8"))
    strategies = {
        "A_no_chunk": lambda: no_chunk(articles),
        "B_recursive_1000": lambda: recursive_char(articles, 1000, 200),
        "C_sentence": lambda: sentence_based(articles, 1000),
        "D_semantic": lambda: semantic_chunk(articles, 0.3),
        "E_parent_child": lambda: parent_child(articles),
    }
    
    all_results = {}
    for name, fn in strategies.items():
        print(f"\n{'='*50}")
        print(f"策略: {name}")
        
        # 1. 分块
        t0 = time.time()
        docs, metas = fn()
        print(f"  分块数: {len(docs)}, 耗时: {time.time()-t0:.1f}s")
        
        # 2. 建临时 ChromaDB 索引 (内存模式, 用完即弃)
        chroma = chromadb.Client()
        # 批量获取 embedding
        t0 = time.time()
        batch, all_embs = 100, []
        for i in range(0, len(docs), batch):
            all_embs.extend(get_embeddings(docs[i:i+batch]))
        print(f"  Embedding 耗时: {time.time()-t0:.1f}s")
        
        col = chroma.create_collection(name=name)
        ids = [f"{name}_{i}" for i in range(len(docs))]
        # ChromaDB 限每批 5461 条
        for i in range(0, len(ids), 5000):
            end = min(i+5000, len(ids))
            col.add(ids=ids[i:end], embeddings=all_embs[i:end],
                    documents=docs[i:end], metadatas=metas[i:end])
        
        # 3. 对每条查询检索 + LLM 打分
        strategy_scores = []
        for q in QUERIES:
            q_emb = get_embeddings([q])[0]
            results = col.query(query_embeddings=[q_emb], n_results=10)
            
            unique_articles = set()
            for j in range(len(results["documents"][0])):
                unique_articles.add(results["metadatas"][0][j]["article_id"])
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                docs_to_score = results["documents"][0]
                scores = list(executor.map(lambda d: judge_relevance(q, d), docs_to_score))
            
            strategy_scores.append({
                "query": q,
                "avg_relevance": sum(scores)/len(scores) if scores else 0,
                "precision": sum(1 for s in scores if s>=1)/len(scores),
                "noise_rate": sum(1 for s in scores if s==0)/len(scores),
                "unique_articles": len(unique_articles),
                "scores": scores,
            })
            print(f"  [{q[:15]}...] avg={strategy_scores[-1]['avg_relevance']:.2f} prec={strategy_scores[-1]['precision']:.0%}")
        
        all_results[name] = {
            "chunk_count": len(docs),
            "queries": strategy_scores,
            "avg_relevance": sum(q["avg_relevance"] for q in strategy_scores)/len(strategy_scores),
            "avg_precision": sum(q["precision"] for q in strategy_scores)/len(strategy_scores),
            "avg_noise": sum(q["noise_rate"] for q in strategy_scores)/len(strategy_scores),
        }
    
    # 保存完整结果
    out = RESULTS_DIR / "chunk_experiment.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 打印汇总表
    print(f"\n{'='*70}")
    print(f"{'策略':<20} {'分块数':>8} {'Avg Rel':>10} {'Precision':>10} {'Noise':>10}")
    print(f"{'-'*70}")
    for name, r in all_results.items():
        print(f"{name:<20} {r['chunk_count']:>8} {r['avg_relevance']:>10.2f} {r['avg_precision']:>9.0%} {r['avg_noise']:>9.0%}")
    print(f"\n结果已保存: {out}")

if __name__ == "__main__":
    run_experiment()
