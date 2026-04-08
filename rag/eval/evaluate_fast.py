"""RAG 分块策略对比实验 (Fast Version)。
5 种策略 × 7 条查询，用 OpenAI API 做 embedding + LLM-as-Judge 评分。
用 asyncio 加速打分。
"""
print("Importing json, time, etc.")
import json, time, re, os, sys, asyncio
from pathlib import Path
print("Importing openai")
from openai import OpenAI, AsyncOpenAI
print("Importing langchain")
from langchain_text_splitters import RecursiveCharacterTextSplitter
print("Importing chromadb")
import chromadb
print("Done importing")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_PATH = Path(__file__).parent / "results" / "subset_100.json"
RESULTS_DIR = Path(__file__).parent / "results"
client_ai = OpenAI()
aclient_ai = AsyncOpenAI()

# ============ 分块策略定义 ============

def no_chunk(articles):
    docs, metas = [], []
    for a in articles:
        docs.append(f"{a['title']}\n{a['summary']}")
        metas.append({"article_id": a["id"], "source": a["source_name"]})
    return docs, metas

def recursive_char(articles, size=1000, overlap=200):
    sp = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    docs, metas = [], []
    for a in articles:
        for chunk in sp.split_text(a["raw_content"]):
            docs.append(chunk)
            metas.append({"article_id": a["id"], "source": a["source_name"]})
    return docs, metas

def sentence_based(articles, max_size=1000):
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
    docs, metas = [], []
    for a in articles:
        sents = [s for s in re.split(r'(?<=[。！？.!?\n])', a["raw_content"]) if s.strip()]
        if len(sents) <= 2:
            docs.append(a["raw_content"])
            metas.append({"article_id": a["id"], "source": a["source_name"]})
            continue
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
    sp_parent = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=0)
    sp_child = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=100)
    docs, metas = [], []
    for a in articles:
        parents = sp_parent.split_text(a["raw_content"])
        for pi, parent in enumerate(parents):
            children = sp_child.split_text(parent)
            for child in children:
                docs.append(child)
                metas.append({"article_id": a["id"], "source": a["source_name"], "parent_text": parent[:800]})
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

async def judge_relevance_async(sem, query, doc_text):
    async with sem:
        try:
            resp = await aclient_ai.chat.completions.create(
                model="openai/gpt-4.1-mini",
                messages=[{"role":"user", "content": f"判断以下检索结果与查询的相关性。\n查询: {query}\n检索结果: {doc_text[:500]}\n只回答一个数字:\n2 = 高度相关\n1 = 部分相关\n0 = 不相关"}],
                temperature=0, max_tokens=20
            )
            return int(resp.choices[0].message.content.strip()[0])
        except Exception as e:
            return 0

# ============ 实验主流程 ============

async def main():
    articles = json.loads(DATA_PATH.read_text("utf-8"))
    strategies = {
        "A_no_chunk": lambda: no_chunk(articles),
        "B_recursive_1000": lambda: recursive_char(articles, 1000, 200),
        "C_sentence": lambda: sentence_based(articles, 1000),
        "E_parent_child": lambda: parent_child(articles),
        "D_semantic": lambda: semantic_chunk(articles, 0.3),  # moved last due to API cost
    }
    
    all_results = {}
    sem = asyncio.Semaphore(15)  # Max 15 concurrent API calls
    
    for name, fn in strategies.items():
        print(f"\n{'='*50}\n策略: {name}")
        t0 = time.time()
        docs, metas = fn()
        print(f"  分块数: {len(docs)}, 耗时: {time.time()-t0:.1f}s")
        
        chroma = chromadb.Client()
        t0 = time.time()
        batch, all_embs = 100, []
        for i in range(0, len(docs), batch):
            all_embs.extend(get_embeddings(docs[i:i+batch]))
        print(f"  Embedding 耗时: {time.time()-t0:.1f}s")
        
        col = chroma.create_collection(name=name)
        ids = [f"{name}_{i}" for i in range(len(docs))]
        for i in range(0, len(ids), 5000):
            end = min(i+5000, len(ids))
            col.add(ids=ids[i:end], embeddings=all_embs[i:end], documents=docs[i:end], metadatas=metas[i:end])
        
        strategy_scores = []
        for q in QUERIES:
            q_emb = get_embeddings([q])[0]
            results = col.query(query_embeddings=[q_emb], n_results=10)
            unique_articles = set(results["metadatas"][0][j]["article_id"] for j in range(len(results["documents"][0])))
            
            tasks = [judge_relevance_async(sem, q, d) for d in results["documents"][0]]
            scores = await asyncio.gather(*tasks)
            
            if not scores: scores = [0]
            strategy_scores.append({
                "query": q, "avg_relevance": sum(scores)/len(scores),
                "precision": sum(1 for s in scores if s>=1)/len(scores),
                "noise_rate": sum(1 for s in scores if s==0)/len(scores),
                "unique_articles": len(unique_articles), "scores": scores,
            })
            print(f"  [{q[:15]}...] avg={strategy_scores[-1]['avg_relevance']:.2f} prec={strategy_scores[-1]['precision']:.0%}")
        
        all_results[name] = {
            "chunk_count": len(docs), "queries": strategy_scores,
            "avg_relevance": sum(q["avg_relevance"] for q in strategy_scores)/len(strategy_scores),
            "avg_precision": sum(q["precision"] for q in strategy_scores)/len(strategy_scores),
            "avg_noise": sum(q["noise_rate"] for q in strategy_scores)/len(strategy_scores),
        }
    
    out = RESULTS_DIR / "chunk_experiment.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}\n{'策略':<20} {'分块数':>8} {'Avg Rel':>10} {'Precision':>10} {'Noise':>10}\n{'-'*70}")
    for name, r in all_results.items():
        print(f"{name:<20} {r['chunk_count']:>8} {r['avg_relevance']:>10.2f} {r['avg_precision']:>9.0%} {r['avg_noise']:>9.0%}")
    print(f"\n结果已保存: {out}")

if __name__ == "__main__":
    asyncio.run(main())
