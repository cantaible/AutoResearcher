"""构建向量索引：读取文章 → 清洗 → embedding → 存入 ChromaDB。"""
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    VECTORDB_DIR, COLLECTION_NAME, EMBEDDING_MODEL, 
    CHUNKS_COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
)
from scripts.db import fetch_all_articles
from scripts.clean_html import clean_html


def parse_timestamp(published_at: str) -> int:
    """把 '2026-04-04T09:01:00Z' 格式的时间转成 Unix 时间戳整数。
    ChromaDB 的 where 过滤只支持数值比较，所以时间必须存为整数。"""
    if not published_at:
        return 0
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return 0


def build():
    # 第一步：从 MariaDB 读取所有文章
    print("1/4 从 MariaDB 读取文章...")
    articles = fetch_all_articles()
    print(f"    共 {len(articles)} 篇")

    # 第二步：准备写入 ChromaDB 的两组数据 (短摘要 + 正文分块)
    ids, documents, metadatas = [], [], []
    chunk_ids, chunk_docs, chunk_metas = [], [], []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    for art in articles:
        title = art["title"] or ""
        summary = clean_html(art["summary"] or "")
        doc_text = f"{title}\n{summary}" if summary else title

        # 清洗全量正文
        raw_text = clean_html(art.get("raw_content") or "")
        preview = raw_text[:500] if raw_text else ""

        ids.append(f"article_{art['id']}")
        documents.append(doc_text)
        metadatas.append({
            "article_id": int(art["id"]),
            "category": art["category"] or "UNCATEGORIZED",
            "source_name": art["source_name"] or "",
            "published_ts": parse_timestamp(art["published_at"]),
            "preview": preview,
        })

        # 为这篇长文生成多个 chunk
        chunks = splitter.split_text(raw_text)
        for idx, chunk in enumerate(chunks):
            chunk_ids.append(f"chunk_{art['id']}_{idx}")
            chunk_docs.append(chunk)
            chunk_metas.append({
                "article_id": int(art["id"]),
                "category": art["category"] or "UNCATEGORIZED",
                "published_ts": parse_timestamp(art["published_at"])
            })

    print(f"    共 {len(chunk_ids)} 个正文分块")

    # 第三步：用 MPS GPU 批量计算 embedding，支持断点续跑
    CACHE_DIR = VECTORDB_DIR / "cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    art_cache = CACHE_DIR / "art_embeddings.npy"
    chunk_cache = CACHE_DIR / "chunk_embeddings.npy"

    print("2/5 加载 embedding 模型 (MPS GPU 加速)...")
    model = SentenceTransformer(EMBEDDING_MODEL, device="mps")
    t0 = time.time()

    # 文章摘要 embedding（有缓存则跳过）
    if art_cache.exists():
        print("3/5 从缓存加载摘要 embedding...")
        art_embeddings = np.load(art_cache)
    else:
        print("3/5 计算摘要 embedding...")
        art_embeddings = model.encode(documents, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
        np.save(art_cache, art_embeddings)
        torch.mps.empty_cache()  # 释放 GPU 显存，给 chunk 留空间

    # 正文分块 embedding：用 CPU 避免 MPS 显存不足卡死，分组+缓存
    if chunk_cache.exists():
        print("   从缓存加载分块 embedding...")
        chunk_embeddings = np.load(chunk_cache)
    else:
        # MPS 16GB 跑不动长文本 chunk，切换到 CPU
        model = model.to("cpu")
        torch.mps.empty_cache()
        group_size = 5000
        groups = list(range(0, len(chunk_docs), group_size))
        all_parts = []
        for gi, start in enumerate(groups):
            part_cache = CACHE_DIR / f"chunk_part_{gi}.npy"
            end = min(start + group_size, len(chunk_docs))
            if part_cache.exists():
                print(f"   分组 {gi+1}/{len(groups)} 从缓存加载")
                all_parts.append(np.load(part_cache))
            else:
                print(f"   分组 {gi+1}/{len(groups)}: chunk {start}-{end}")
                part = model.encode(chunk_docs[start:end], batch_size=128,
                    show_progress_bar=True, normalize_embeddings=True)
                np.save(part_cache, part)
                all_parts.append(part)
        chunk_embeddings = np.concatenate(all_parts)
        np.save(chunk_cache, chunk_embeddings)

    print(f"    embedding 完成，耗时 {time.time()-t0:.1f}s")
    del model

    # 第四步：写入 ChromaDB
    VECTORDB_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
    existing = [c.name for c in client.list_collections()]
    for c_name in [COLLECTION_NAME, CHUNKS_COLLECTION_NAME]:
        if c_name in existing:
            client.delete_collection(c_name)
    client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
    collection = client.create_collection(name=COLLECTION_NAME)
    chunks_col = client.create_collection(name=CHUNKS_COLLECTION_NAME)

    print("4/5 写入摘要向量...")
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(ids=ids[i:end], embeddings=art_embeddings[i:end].tolist(), documents=documents[i:end], metadatas=metadatas[i:end])

    print("5/5 写入正文分块向量...")
    for i in range(0, len(chunk_ids), batch_size):
        end = min(i + batch_size, len(chunk_ids))
        chunks_col.add(ids=chunk_ids[i:end], embeddings=chunk_embeddings[i:end].tolist(), documents=chunk_docs[i:end], metadatas=chunk_metas[i:end])
        if end % 5000 == 0 or end == len(chunk_ids):
            print(f"    chunk {end}/{len(chunk_ids)}")

    elapsed = time.time() - t0
    print(f"✅ 完成！ {collection.count()} 篇摘要 + {chunks_col.count()} 个分块，总耗时 {elapsed:.1f}s")

if __name__ == "__main__":
    # 安全检查：索引已存在时，必须加 --force 才能重建
    if VECTORDB_DIR.exists() and any(VECTORDB_DIR.iterdir()):
        if "--force" not in sys.argv:
            print(f"⚠️  索引已存在: {VECTORDB_DIR}")
            print(f"   重建需要约 20 分钟，如确认要重建，请运行:")
            print(f"   python scripts/build_index.py --force")
            sys.exit(1)
        print("⚠️  --force 已指定，将删除旧索引并重建...")
    build()
