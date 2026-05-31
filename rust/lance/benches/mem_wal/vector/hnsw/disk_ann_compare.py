#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Disk-ANN comparison: Lance on-disk IVF_HNSW_SQ (the flushed-memtable index)
vs DiskANN vs FAISS, all backed by local NVMe.

All indexes live under --base (an NVMe mount). The Lance index is served fully
cached in memory (large index_cache_size_bytes). Dataset: dbpedia-1M OpenAI
embeddings (1536-d, cosine). For each corpus size we sweep each system's search
parameter and report recall@10 vs p50/p99 latency and QPS, so systems can be
compared at matched recall.

Run one size:  python disk_ann_compare.py --rows 100000 --base /mnt/nvme/anncmp
"""
import argparse, json, os, time, struct
import numpy as np

K = 10
NUM_QUERIES = 1000
SEED = 42
DIM = 1536
HF_TREE = "https://huggingface.co/api/datasets/KShivendu/dbpedia-entities-openai-1M/tree/main/data"
HF_BASE = "https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M/resolve/main/"


def load_corpus(cache_dir, needed):
    """Download dbpedia parquet shards to NVMe cache, return (needed, DIM) f32."""
    import requests, pyarrow.parquet as pq
    os.makedirs(cache_dir, exist_ok=True)
    shards = sorted(
        e["path"] for e in requests.get(HF_TREE, timeout=60).json()
        if e["type"] == "file" and e["path"].endswith(".parquet")
    )
    out = np.empty((needed, DIM), dtype=np.float32)
    n = 0
    for rel in shards:
        if n >= needed:
            break
        local = os.path.join(cache_dir, os.path.basename(rel))
        if not os.path.exists(local):
            r = requests.get(HF_BASE + rel, timeout=600)
            r.raise_for_status()
            with open(local, "wb") as f:
                f.write(r.content)
        col = pq.read_table(local, columns=["openai"]).column("openai")
        arr = np.stack(col.to_pylist()).astype(np.float32)
        take = min(len(arr), needed - n)
        out[n:n + take] = arr[:take]
        n += take
        print(f"  shard {os.path.basename(rel)} -> {take} rows (cum {n})", flush=True)
    assert n == needed, f"only got {n}/{needed}"
    return out


def normalize(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return x / nrm


def ground_truth(corpus, queries):
    import faiss
    idx = faiss.IndexFlatIP(DIM)
    idx.add(corpus)
    _, ids = idx.search(queries, K)
    return ids


def recall_at_k(gt, got):
    hits = sum(len(set(g.tolist()) & set(r.tolist())) for g, r in zip(gt, got))
    return hits / (len(gt) * K)


def latency_qps(query_fn, queries, repeats=3):
    # single-query latency percentiles (serial) over repeats
    lat = []
    for _ in range(repeats):
        for q in queries:
            t = time.perf_counter()
            query_fn(q)
            lat.append((time.perf_counter() - t) * 1e6)
    lat.sort()
    p50 = lat[len(lat) // 2]
    p99 = lat[int(len(lat) * 0.99)]
    qps = 1e6 / (sum(lat) / len(lat))
    return p50, p99, qps


# ---------------- Lance ----------------
def bench_lance(base, corpus, queries, gt, params):
    import lance, pyarrow as pa
    uri = os.path.join(base, "lance_ds")
    ids = pa.array(np.arange(len(corpus), dtype=np.int64))
    vecs = pa.FixedSizeListArray.from_arrays(pa.array(corpus.reshape(-1), type=pa.float32()), DIM)
    tbl = pa.table({"id": ids, "vec": vecs})
    import shutil
    shutil.rmtree(uri, ignore_errors=True)
    ds = lance.write_dataset(tbl, uri, mode="overwrite")
    nlist = params.get("nlist", max(1, int(np.sqrt(len(corpus)))))
    t = time.perf_counter()
    ds.create_index("vec", "IVF_HNSW_SQ", metric="cosine", num_partitions=nlist,
                    m=params.get("m", 20), ef_construction=params.get("ef_construction", 150))
    build_s = time.perf_counter() - t
    # Fully cache the index in memory.
    ds = lance.dataset(uri, index_cache_size_bytes=32 * 1024**3)

    def make_q(ef, nprobes):
        def q(vec):
            return ds.to_table(nearest={"column": "vec", "q": vec, "k": K,
                                        "nprobes": nprobes, "ef": ef},
                               columns=["id"]).column("id").to_numpy()
        return q

    rows = []
    for ef in params.get("ef_search", [16, 32, 64, 128, 256]):
        nprobes = params.get("nprobes", nlist)
        qf = make_q(ef, nprobes)
        for vec in queries[:50]:  # warm
            qf(vec)
        got = np.stack([qf(v) for v in queries])
        rec = recall_at_k(gt, got)
        p50, p99, qps = latency_qps(qf, queries)
        rows.append({"ef": ef, "nprobes": nprobes, "recall": rec, "p50_us": p50, "p99_us": p99, "qps": qps})
        print(f"  lance ef={ef} nprobes={nprobes} recall={rec:.4f} p50={p50:.0f}us qps={qps:.0f}", flush=True)
    return {"build_s": build_s, "sweep": rows}


# ---------------- FAISS ----------------
def bench_faiss(base, corpus, queries, gt, params):
    import faiss
    M = params.get("m", 32)
    index = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = params.get("ef_construction", 200)
    t = time.perf_counter()
    index.add(corpus)
    build_s = time.perf_counter() - t
    faiss.write_index(index, os.path.join(base, "faiss_hnsw.index"))

    def make_q(ef):
        def q(vec):
            index.hnsw.efSearch = ef
            _, ids = index.search(vec.reshape(1, -1), K)
            return ids[0]
        return q

    rows = []
    for ef in params.get("ef_search", [16, 32, 64, 128, 256]):
        qf = make_q(ef)
        for vec in queries[:50]:
            qf(vec)
        got = np.stack([qf(v) for v in queries])
        rec = recall_at_k(gt, got)
        p50, p99, qps = latency_qps(qf, queries)
        rows.append({"ef": ef, "recall": rec, "p50_us": p50, "p99_us": p99, "qps": qps})
        print(f"  faiss ef={ef} recall={rec:.4f} p50={p50:.0f}us qps={qps:.0f}", flush=True)
    return {"build_s": build_s, "sweep": rows}


# ---------------- DiskANN ----------------
def bench_diskann(base, corpus, queries, gt, params):
    import diskannpy as dap
    idx_dir = os.path.join(base, "diskann")
    os.makedirs(idx_dir, exist_ok=True)
    t = time.perf_counter()
    dap.build_memory_index(
        data=corpus, distance_metric="cosine",
        index_directory=idx_dir, index_prefix="ann",
        complexity=params.get("ef_construction", 150),
        graph_degree=params.get("m", 32) * 2,
        num_threads=0, alpha=1.2, use_pq_build=False, num_pq_bytes=0,
    )
    build_s = time.perf_counter() - t
    idx = dap.StaticMemoryIndex(index_directory=idx_dir, index_prefix="ann",
                                num_threads=0, initial_search_complexity=256)

    def make_q(L):
        def q(vec):
            return idx.search(vec, k_neighbors=K, complexity=L).identifiers
        return q

    rows = []
    for L in params.get("ef_search", [16, 32, 64, 128, 256]):
        qf = make_q(L)
        for vec in queries[:50]:
            qf(vec)
        got = np.stack([qf(v) for v in queries])
        rec = recall_at_k(gt, got)
        p50, p99, qps = latency_qps(qf, queries)
        rows.append({"L": L, "recall": rec, "p50_us": p50, "p99_us": p99, "qps": qps})
        print(f"  diskann L={L} recall={rec:.4f} p50={p50:.0f}us qps={qps:.0f}", flush=True)
    return {"build_s": build_s, "sweep": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--systems", default="lance,faiss,diskann")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.makedirs(args.base, exist_ok=True)
    cache = os.path.join(args.base, "dbpedia_cache")
    raw = load_corpus(cache, args.rows + NUM_QUERIES if args.rows < 1_000_000 else args.rows)
    corpus = normalize(raw[:args.rows])
    rng = np.random.default_rng(SEED)
    qidx = rng.choice(args.rows, size=NUM_QUERIES, replace=False)
    queries = corpus[qidx].copy()
    print(f"corpus={len(corpus)} queries={len(queries)} dim={DIM}", flush=True)
    gt = ground_truth(corpus, queries)

    params = {"m": 20, "ef_construction": 150, "ef_search": [16, 32, 64, 128, 256]}
    results = {"rows": args.rows}
    for sys in args.systems.split(","):
        print(f"=== {sys} (rows={args.rows}) ===", flush=True)
        fn = {"lance": bench_lance, "faiss": bench_faiss, "diskann": bench_diskann}[sys]
        try:
            results[sys] = fn(args.base, corpus, queries, gt, params)
        except Exception as e:
            print(f"  {sys} FAILED: {e}", flush=True)
            results[sys] = {"error": str(e)}

    out = args.out or os.path.join(args.base, f"result_{args.rows}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"=== wrote {out} ===", flush=True)


if __name__ == "__main__":
    main()
