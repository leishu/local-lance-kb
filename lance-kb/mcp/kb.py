#!/usr/bin/env python3
"""lance-kb 数据层：本地知识库的入库、切块、向量化、混合检索与统计。

存哪：LanceDB（列式 + 向量索引 + 全文索引）落在 data/lancedb/，一个目录就是整个库，
      复制/备份/搬家都只是拷目录。
怎么存：一个文件 = docs 表一行；切出来的片段 = chunks 表一行，带出处（路径、标题、序号）。
怎么查：向量召回 + 全文召回各自排序，再用 RRF 融合（见 search()），
      纯向量在中文专有名词上不灵，纯全文在近义改写上不灵，两个一起才够用。

命令行（等价于 MCP 工具，方便脚本和定时任务）：
    python3 mcp/kb.py add <路径>... [--tags a,b] [--force]
    python3 mcp/kb.py note --name 标题 --text-file x.md
    python3 mcp/kb.py search -q "怎么配 MCP" [-k 8] [--mode hybrid]
    python3 mcp/kb.py stats
    python3 mcp/kb.py docs [--limit 50]
    python3 mcp/kb.py rm --doc-id <id> | --path <前缀> | --pattern 'glob' | --all
    python3 mcp/kb.py sync
    python3 mcp/kb.py reindex [--doc-id <id>]
    python3 mcp/kb.py export-dev
"""

from __future__ import annotations

import argparse
import fnmatch
import glob as _glob
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import embedders  # noqa: E402
import loaders  # noqa: E402

SCHEMA_VERSION = 1
HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN_ROOT = os.path.dirname(HERE)
KB_HOME = os.path.abspath(os.environ.get("LANCE_KB_HOME")
                          or os.path.join(PLUGIN_ROOT, "data"))
LANCE_DIR = os.path.join(KB_HOME, "lancedb")
CONFIG_PATH = os.path.join(KB_HOME, "config.json")

T_CHUNKS = "chunks"
T_DOCS = "docs"

DEFAULT_CHUNK = 700
DEFAULT_OVERLAP = 120
RRF_K = 60


class NeedReindex(RuntimeError):
    """向量维度与当前嵌入后端不一致。"""


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S")


def load_config() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        cfg = {}
    cfg.setdefault("schema_version", SCHEMA_VERSION)
    cfg.setdefault("chunk", {"size": DEFAULT_CHUNK, "overlap": DEFAULT_OVERLAP})
    cfg.setdefault("sources", [])
    return cfg


def save_config(cfg: dict) -> None:
    os.makedirs(KB_HOME, exist_ok=True)
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, CONFIG_PATH)


# --------------------------------------------------------------------------- #
# LanceDB 连接与表结构
# --------------------------------------------------------------------------- #

_STATE: dict = {}


def db():
    if "db" not in _STATE:
        import lancedb
        os.makedirs(LANCE_DIR, exist_ok=True)
        _STATE["db"] = lancedb.connect(LANCE_DIR)
    return _STATE["db"]


def _table_names(handle) -> list[str]:
    for attr in ("list_tables", "table_names"):
        fn = getattr(handle, attr, None)
        if fn is None:
            continue
        try:
            out = fn()
        except Exception:
            continue
        if isinstance(out, list):
            return [str(x) for x in out]
        tables = getattr(out, "tables", None)
        if tables is not None:
            return [str(x) for x in tables]
    return []


def _schema(dim: int):
    import pyarrow as pa
    return pa.schema([
        pa.field("chunk_id", pa.string()),
        pa.field("doc_id", pa.string()),
        pa.field("doc_name", pa.string()),
        pa.field("doc_path", pa.string()),
        pa.field("doc_type", pa.string()),
        pa.field("source", pa.string()),
        pa.field("heading", pa.string()),
        pa.field("text", pa.string()),
        pa.field("search_text", pa.string()),
        pa.field("ordinal", pa.int32()),
        pa.field("n_chars", pa.int32()),
        pa.field("tags", pa.string()),
        pa.field("created_at", pa.string()),
        pa.field("doc_mtime", pa.float64()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])


DOCS_SCHEMA_FIELDS = [
    "doc_id", "doc_name", "doc_path", "doc_type", "source", "size", "mtime",
    "content_hash", "n_chunks", "n_chars", "tags", "added_at", "updated_at",
]


def _docs_schema():
    import pyarrow as pa
    return pa.schema([
        pa.field("doc_id", pa.string()),
        pa.field("doc_name", pa.string()),
        pa.field("doc_path", pa.string()),
        pa.field("doc_type", pa.string()),
        pa.field("source", pa.string()),
        pa.field("size", pa.int64()),
        pa.field("mtime", pa.float64()),
        pa.field("content_hash", pa.string()),
        pa.field("n_chunks", pa.int32()),
        pa.field("n_chars", pa.int32()),
        pa.field("tags", pa.string()),
        pa.field("added_at", pa.string()),
        pa.field("updated_at", pa.string()),
    ])


def embedder() -> embedders.Embedder:
    """解析嵌入后端，并把它的身份记进 config —— 换后端要 reindex，靠这个判断。"""
    if "embedder" not in _STATE:
        cfg = load_config()
        want = cfg.get("embedder") or {}
        prefer = os.environ.get("LANCE_KB_EMBED") or "auto"
        emb = embedders.resolve(prefer)
        known = want.get("name")
        if known and known != "auto" and prefer == "auto" and known != emb.name:
            # config 记的是 fastembed，但这台机器上没装 → 说明换了环境，
            # 维度大概率对不上，交给 ensure() 报警，不在这里静默改配置
            pass
        _STATE["embedder"] = emb
        if not want or want.get("dim") != emb.dim or want.get("name") != emb.name:
            cfg["embedder"] = emb.info()
            save_config(cfg)
    return _STATE["embedder"]


def _open(name: str):
    handle = db()
    if name not in _table_names(handle):
        return None
    return handle.open_table(name)


def ensure_tables() -> tuple:
    """确保两张表存在，并校验向量维度。维度对不上时抛 NeedReindex，绝不写脏数据。"""
    handle = db()
    emb = embedder()
    names = _table_names(handle)
    chunks = _open(T_CHUNKS)
    if chunks is None:
        chunks = handle.create_table(T_CHUNKS, schema=_schema(emb.dim))
    else:
        dim = _vector_dim(chunks)
        if dim and dim != emb.dim:
            # 空库直接换维度重来 —— 用户常是先装插件、后配 ollama/fastembed，
            # 这时逼他 reindex 一个空库纯属添堵。
            try:
                empty = chunks.count_rows() == 0
            except Exception:
                empty = False
            if empty:
                handle.drop_table(T_CHUNKS)
                chunks = handle.create_table(T_CHUNKS, schema=_schema(emb.dim))
            else:
                raise NeedReindex(
                    f"库里的向量是 {dim} 维，当前嵌入后端 {emb.name}/{emb.model} 是 {emb.dim} 维。"
                    f"换后端后必须重建：kb_reindex（或 LANCE_KB_EMBED=hash 回到原来的后端）。")
    docs = _open(T_DOCS)
    if docs is None:
        docs = handle.create_table(T_DOCS, schema=_docs_schema())
    return chunks, docs


def _vector_dim(tbl) -> int | None:
    try:
        for field in tbl.schema:
            if field.name == "vector":
                return field.type.list_size
    except Exception:
        pass
    return None


def _rows(tbl, columns=None, where=None, limit=None) -> list[dict]:
    """读表：优先走 Lance 扫描器（可下推列裁剪/过滤/limit），不行再退回全表。"""
    try:
        ds = tbl.to_lance()
        kwargs = {}
        if columns:
            kwargs["columns"] = list(columns)
        if where:
            kwargs["filter"] = where
        if limit:
            kwargs["limit"] = int(limit)
        return ds.to_table(**kwargs).to_pylist()
    except Exception:
        rows = tbl.to_arrow().to_pylist()
        if where:
            rows = [r for r in rows if _py_match(r, where)]
        if columns:
            rows = [{k: r.get(k) for k in columns} for r in rows]
        return rows[: int(limit)] if limit else rows


def _py_match(row: dict, where: str) -> bool:
    """极简 where 兜底：只支持 `field = 'v'` 与 AND 连接。"""
    for part in re.split(r"\s+AND\s+", where, flags=re.I):
        m = re.match(r"\s*(\w+)\s*=\s*'((?:[^']|'')*)'\s*$", part)
        if not m:
            return True
        if str(row.get(m.group(1))) != m.group(2).replace("''", "'"):
            return False
    return True


def _q(value: str) -> str:
    """拼 where 子句用的字面量，单引号转义成两个。"""
    return "'" + str(value).replace("'", "''") + "'"


# --------------------------------------------------------------------------- #
# 入库
# --------------------------------------------------------------------------- #

def _sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def doc_id_for(path: str) -> str:
    return "f" + _sha1(os.path.abspath(path).encode("utf-8"))[:15]


def note_id_for(name: str, text: str) -> str:
    return "n" + _sha1((name + "\x00" + text).encode("utf-8"))[:15]


def _meta_index(docs) -> dict:
    return {r["doc_id"]: r for r in _rows(docs)}


def _embed_texts(emb, items: list[dict], batch: int = 64):
    """分批量向量化；标题拼进正文再嵌入，让「小节标题」也参与语义。"""
    vecs = []
    for i in range(0, len(items), batch):
        part = items[i:i + batch]
        texts = [((it["heading"] + "\n" if it["heading"] else "") + it["text"])[:3000]
                 for it in part]
        vecs.extend(emb.encode(texts))
    return vecs


def _write_doc(chunks_tbl, docs_tbl, doc: dict, chunks: list[dict], emb) -> None:
    """先删旧片段再写新片段，保证同一文档不会新旧混杂。"""
    import pyarrow as pa

    chunks_tbl.delete(f"doc_id = {_q(doc['doc_id'])}")
    if chunks:
        vecs = _embed_texts(emb, chunks)
        rows = []
        for n, (c, v) in enumerate(zip(chunks, vecs)):
            rows.append({
                "chunk_id": f"{doc['doc_id']}#{n:04d}",
                "doc_id": doc["doc_id"],
                "doc_name": doc["doc_name"],
                "doc_path": doc["doc_path"],
                "doc_type": doc["doc_type"],
                "source": doc["source"],
                "heading": c["heading"] or "",
                "text": c["text"],
                "search_text": (c["heading"] + "\n" if c["heading"] else "") + c["text"],
                "ordinal": n,
                "n_chars": len(c["text"]),
                "tags": doc["tags"],
                "created_at": now_iso(),
                "doc_mtime": float(doc["mtime"] or 0.0),
                "vector": v,
            })
        tbl = pa.Table.from_pylist(rows, schema=_schema(emb.dim))
        chunks_tbl.merge_insert("chunk_id") \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .execute(tbl)

    docs_tbl.delete(f"doc_id = {_q(doc['doc_id'])}")
    docs_tbl.add(pa.Table.from_pylist([doc], schema=_docs_schema()))


def _retag(chunks_tbl, docs_tbl, doc: dict, tags: str) -> None:
    """内容没变但要改标签：只改 tags 列，不重算向量（重嵌入太贵且没必要）。"""
    import pyarrow as pa

    updated = dict(doc)
    updated["tags"] = tags
    updated["updated_at"] = now_iso()
    docs_tbl.delete(f"doc_id = {_q(doc['doc_id'])}")
    docs_tbl.add(pa.Table.from_pylist([{k: updated.get(k) for k in DOCS_SCHEMA_FIELDS}],
                                      schema=_docs_schema()))
    try:
        chunks_tbl.update(where=f"doc_id = {_q(doc['doc_id'])}", values={"tags": tags})
    except Exception:
        # 老版本 update 不支持 where 关键字，退回逐行 SQL 形式的 update
        try:
            chunks_tbl.update(f"doc_id = {_q(doc['doc_id'])}", {"tags": tags})
        except Exception:
            pass


def add_paths(paths, tags=None, force=False, recursive=True, chunk_size=None,
              overlap=None, log=None, record_source=True, build_fts=True) -> dict:
    log = log or (lambda m: None)
    cfg = load_config()
    chunk_size = int(chunk_size or cfg["chunk"].get("size") or DEFAULT_CHUNK)
    overlap = int(overlap if overlap is not None else cfg["chunk"].get("overlap", DEFAULT_OVERLAP))
    tag_str = _norm_tags(tags)
    emb = embedder()
    chunks_tbl, docs_tbl = ensure_tables()
    index = _meta_index(docs_tbl)

    report = {"added": [], "updated": [], "skipped": [], "failed": [], "retagged": [],
              "chunks": 0, "chars": 0, "embedder": emb.describe(), "elapsed_sec": 0}
    started = time.time()
    files = list(loaders.iter_files(paths, recursive=recursive))
    if not files:
        report["failed"].append({"path": ", ".join(map(str, paths)),
                                 "error": "没有匹配到可读文件（支持 md/txt/pdf/docx/html/csv/json/代码等）"})
        return report

    for path in files:
        try:
            stat = os.stat(path)
            with open(path, "rb") as fh:
                digest = _sha1(fh.read())
            doc_id = doc_id_for(path)
            old = index.get(doc_id)
            if old and old.get("content_hash") == digest and not force:
                if tag_str and tag_str != (old.get("tags") or ""):
                    _retag(chunks_tbl, docs_tbl, old, tag_str)
                    index[doc_id] = dict(old, tags=tag_str)
                    report["retagged"].append({"doc_id": doc_id, "name": old.get("doc_name"),
                                               "tags": tag_str})
                    continue
                report["skipped"].append({"path": path, "reason": "内容未变"})
                continue
            doc_type = loaders.detect_type(path) or "txt"
            blocks = loaders.read_blocks(path, doc_type)
            pieces = loaders.chunk_blocks(blocks, chunk_size, overlap)
            if not pieces:
                report["skipped"].append({"path": path, "reason": "没有可索引的文本"})
                continue
            doc = {
                "doc_id": doc_id,
                "doc_name": os.path.basename(path),
                "doc_path": os.path.abspath(path),
                "doc_type": doc_type,
                "source": "file",
                "size": int(stat.st_size),
                "mtime": float(stat.st_mtime),
                "content_hash": digest,
                "n_chunks": len(pieces),
                "n_chars": sum(len(p["text"]) for p in pieces),
                "tags": tag_str or (old or {}).get("tags", ""),
                "added_at": (old or {}).get("added_at") or now_iso(),
                "updated_at": now_iso(),
            }
            _write_doc(chunks_tbl, docs_tbl, doc, pieces, emb)
            index[doc_id] = doc
            report["updated" if old else "added"].append(
                {"doc_id": doc_id, "name": doc["doc_name"], "path": doc["doc_path"],
                 "chunks": len(pieces), "chars": doc["n_chars"]})
            report["chunks"] += len(pieces)
            report["chars"] += doc["n_chars"]
            log(f"+ {doc['doc_name']} ({len(pieces)} 块)")
        except loaders.MissingDep as exc:
            report["failed"].append({"path": path, "error": str(exc)})
        except Exception as exc:  # 单文件失败不拖垮整批
            report["failed"].append({"path": path, "error": f"{type(exc).__name__}: {exc}"})

    if record_source:
        _record_source(cfg, paths, tag_str)
    if build_fts:
        _ensure_fts(chunks_tbl, rebuild=report["chunks"] > 0)
    report["elapsed_sec"] = round(time.time() - started, 2)
    report["total_docs"] = docs_tbl.count_rows()
    return report


def add_note(name: str, text: str, tags=None, chunk_size=None, overlap=None) -> dict:
    cfg = load_config()
    chunk_size = int(chunk_size or cfg["chunk"].get("size") or DEFAULT_CHUNK)
    overlap = int(overlap if overlap is not None else cfg["chunk"].get("overlap", DEFAULT_OVERLAP))
    emb = embedder()
    chunks_tbl, docs_tbl = ensure_tables()
    doc_id = note_id_for(name, text)
    blocks = loaders.text_blocks_for_text(text)
    pieces = loaders.chunk_blocks(blocks, chunk_size, overlap)
    if not pieces:
        return {"error": "文本为空，没有可索引内容"}
    digest = _sha1(text.encode("utf-8"))
    old = _meta_index(docs_tbl).get(doc_id)
    doc = {
        "doc_id": doc_id,
        "doc_name": name,
        "doc_path": f"note://{name}",
        "doc_type": "note",
        "source": "note",
        "size": len(text.encode("utf-8")),
        "mtime": time.time(),
        "content_hash": digest,
        "n_chunks": len(pieces),
        "n_chars": sum(len(p["text"]) for p in pieces),
        "tags": _norm_tags(tags) or (old or {}).get("tags", ""),
        "added_at": (old or {}).get("added_at") or now_iso(),
        "updated_at": now_iso(),
    }
    _write_doc(chunks_tbl, docs_tbl, doc, pieces, emb)
    _ensure_fts(chunks_tbl, rebuild=True)
    return {"doc_id": doc_id, "name": name, "chunks": len(pieces),
            "chars": doc["n_chars"], "embedder": emb.describe()}


def _norm_tags(tags) -> str:
    if not tags:
        return ""
    if isinstance(tags, str):
        items = re.split(r"[,，;；\s]+", tags)
    else:
        items = [str(t) for t in tags]
    out = []
    for t in items:
        t = t.strip().lstrip("#")
        if t and t not in out:
            out.append(t)
    return ",".join(out)


def _record_source(cfg: dict, paths, tags: str) -> None:
    """记住用户喂过哪些目录，kb_sync 才有东西可重扫。"""
    sources = cfg.setdefault("sources", [])
    known = {s.get("path") for s in sources}
    for raw in paths:
        p = os.path.expanduser(str(raw).strip())
        if not p or any(c in p for c in "*?["):
            p = os.path.dirname(os.path.abspath(p or "."))
        ap = os.path.abspath(p)
        if ap in known:
            rec = next(s for s in sources if s.get("path") == ap)
            if tags:
                rec["tags"] = tags
            continue
        sources.append({
            "path": ap,
            "kind": "dir" if os.path.isdir(ap) else "file",
            "tags": tags,
            "added_at": now_iso(),
        })
    save_config(cfg)


# --------------------------------------------------------------------------- #
# 全文索引
# --------------------------------------------------------------------------- #

def _fts_label(tbl) -> str:
    """已建索引时把分词器名字读回来，别在 stats 里显示 "existing" 这种内部状态。"""
    try:
        for idx in tbl.list_indices():
            name = (getattr(idx, "name", "") or "").upper()
            kind = (getattr(idx, "index_type", "") or "").upper()
            if not ("FTS" in name or "FTS" in kind or "INVERTED" in kind
                    or name.startswith("SEARCH_TEXT")):
                continue
            details = getattr(idx, "index_details", None) or {}
            if isinstance(details, dict):
                base = details.get("base_tokenizer")
                if base == "ngram":
                    lo = details.get("min_ngram_length")
                    hi = details.get("max_ngram_length")
                    if lo and hi:
                        return f"ngram({lo}-{hi})"
                if base:
                    return str(base)
            return "已建立"
    except Exception:
        pass
    return ""


def _fts_config():
    """全文索引配置。

    LanceDB 0.4x 的 native FTS 只认 simple/raw/whitespace/ngram/icu 这几种分词器，
    没有 jieba（icu 也不支持 Chinese 语言）；而 simple 按空白切词，中文整段会变成
    一个 token，等于搜不到。所以中文场景唯一可用的是 ngram（2~3 字），代价是索引更大。
    """
    from lancedb.index import FTS
    kind = (load_config().get("fts") or os.environ.get("LANCE_KB_FTS") or "ngram").lower()
    if kind in ("simple", "raw", "whitespace"):
        return FTS(base_tokenizer=kind), kind
    return FTS(base_tokenizer="ngram", ngram_min_length=2, ngram_max_length=3), "ngram(2-3)"


def _ensure_fts(tbl, rebuild: bool = True) -> str:
    """给 search_text 建全文索引（标题 + 正文，一个字段，native FTS 不支持多字段）。

    rebuild=False 时只在还没索引时才建 —— 每批入库都重建索引太贵。
    """
    if not rebuild:
        label = _fts_label(tbl)
        if label:
            _STATE["fts"] = label
            return label

    try:
        cfg, label = _fts_config()
        tbl.create_index("search_text", config=cfg)
        _STATE["fts"] = label
        return label
    except ImportError:
        pass
    except Exception as exc:
        # 记下来给 stats 排障用，别让索引建失败变成「悄无声息地只能字面搜」
        _STATE["fts_error"] = f"{type(exc).__name__}: {exc}"

    # 老 API 兜底（lancedb < 0.25）
    for tokenizer in ("ngram", "simple"):
        try:
            tbl.create_fts_index("search_text", replace=True, tokenizer_name=tokenizer)
            _STATE["fts"] = tokenizer
            return tokenizer
        except Exception:
            continue
    _STATE["fts"] = ""
    return ""


# --------------------------------------------------------------------------- #
# 检索
# --------------------------------------------------------------------------- #

def _where(filters: dict | None) -> str:
    filters = filters or {}
    parts = []
    if filters.get("doc_type"):
        types = filters["doc_type"]
        types = [types] if isinstance(types, str) else list(types)
        types = [t for t in types if t]
        if types:
            parts.append("doc_type IN (" + ", ".join(_q(t) for t in types) + ")")
    if filters.get("doc_id"):
        ids = filters["doc_id"]
        ids = [ids] if isinstance(ids, str) else list(ids)
        parts.append("doc_id IN (" + ", ".join(_q(i) for i in ids) + ")")
    if filters.get("path_prefix"):
        # LIKE 前缀 + 单引号转义
        prefix = str(filters["path_prefix"]).replace("'", "''")
        parts.append(f"doc_path LIKE '{prefix}%'")
    if filters.get("path_contains"):
        needle = str(filters["path_contains"]).replace("'", "''")
        parts.append(f"doc_path LIKE '%{needle}%'")
    if filters.get("tags"):
        tags = filters["tags"]
        tags = [tags] if isinstance(tags, str) else list(tags)
        for t in tags:
            if t:
                parts.append(f"tags LIKE {_q('%' + str(t).strip() + '%')}")
    return " AND ".join(parts)


def _vector_search(tbl, vec, k, where: str):
    q = tbl.search(vec, vector_column_name="vector").limit(k)
    try:
        q = q.metric("cosine")
    except Exception:
        pass
    if where:
        try:
            q = q.where(where, prefilter=True)
        except TypeError:
            q = q.where(where)
    rows = q.to_list()
    for r in rows:
        d = r.get("_distance")
        r["similarity"] = None if d is None else max(-1.0, min(1.0, 1.0 - float(d)))
    return rows


def _fts_search(tbl, query, k, where: str):
    try:
        q = tbl.search(query, query_type="fts", fts_columns="search_text").limit(k)
        if where:
            try:
                q = q.where(where, prefilter=True)
            except TypeError:
                q = q.where(where)
        return q.to_list()
    except Exception:
        # 老版本 / 没建索引：退化成 LIKE 扫描（慢，但能出结果）
        try:
            kw = query.replace("'", "''")
            q = tbl.search().where(f"text LIKE '%{kw}%'", prefilter=True).limit(k)
            return q.to_list()
        except Exception:
            return []


def _rrf(lists, k: int, rrf_k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion：只看名次不看分数，天然解决两路分数量纲不同的问题。"""
    merged: dict[str, dict] = {}
    for tag, rows in lists:
        for rank, row in enumerate(rows, 1):
            cid = row.get("chunk_id")
            if not cid:
                continue
            entry = merged.setdefault(cid, {"row": row, "score": 0.0, "ranks": {}})
            entry["score"] += 1.0 / (rrf_k + rank)
            entry["ranks"][tag] = rank
    out = sorted(merged.values(), key=lambda e: -e["score"])[:k]
    for e in out:
        e["row"]["score"] = round(e["score"], 5)
        e["row"]["ranks"] = e["ranks"]
    return [e["row"] for e in out]


def _slim(row: dict, limit: int = 900) -> dict:
    text = row.get("text") or ""
    return {
        "chunk_id": row.get("chunk_id"),
        "doc_id": row.get("doc_id"),
        "doc_name": row.get("doc_name"),
        "doc_path": row.get("doc_path"),
        "doc_type": row.get("doc_type"),
        "heading": row.get("heading") or "",
        "ordinal": row.get("ordinal"),
        "text": text if len(text) <= limit else text[:limit] + " …",
        "truncated": len(text) > limit,
        "n_chars": row.get("n_chars"),
        "score": row.get("score"),
        "ranks": row.get("ranks") or {},
        "similarity": None if row.get("similarity") is None else round(row["similarity"], 4),
        "tags": row.get("tags") or "",
    }


def search(query: str, k: int = 8, mode: str = "hybrid", filters: dict | None = None,
           text_limit: int = 900) -> dict:
    query = (query or "").strip()
    if not query:
        return {"query": "", "hits": [], "message": "查询为空"}
    k = max(1, min(int(k or 8), 50))
    mode = (mode or "hybrid").lower()
    emb = embedder()
    chunks_tbl, docs_tbl = ensure_tables()
    where = _where(filters)
    if docs_tbl.count_rows() == 0:
        return {"query": query, "hits": [], "empty": True,
                "message": "库里还没有文档。先用 kb_add 把文件或目录加进来。"}

    pool = max(k * 4, 20)
    lists = []
    errors = []
    if mode in ("hybrid", "vector"):
        try:
            vec = emb.encode([query])[0]
            lists.append(("vector", _vector_search(chunks_tbl, vec, pool, where)))
        except Exception as exc:
            errors.append(f"向量检索失败：{exc}")
    if mode in ("hybrid", "keyword"):
        rows = _fts_search(chunks_tbl, query, pool, where)
        lists.append(("keyword", rows))
    if not lists:
        return {"query": query, "hits": [], "error": "；".join(errors) or "检索失败"}

    if len(lists) == 1:
        hits = lists[0][1][:k]
        for row in hits:
            row["score"] = 1.0
            row["ranks"] = {lists[0][0]: hits.index(row) + 1}
    else:
        hits = _rrf(lists, k)
    return {
        "query": query,
        "mode": mode if len(lists) > 1 else lists[0][0],
        "k": k,
        "hits": [_slim(h, text_limit) for h in hits],
        "filter": where or None,
        "embedder": emb.describe() if mode != "keyword" else "keyword-only",
        "errors": errors or None,
    }


def get_chunk(chunk_id: str = None, doc_id: str = None, before: int = 1, after: int = 2,
              ordinal: int = None) -> dict:
    chunks_tbl, docs_tbl = ensure_tables()
    if chunk_id:
        rows = _rows(chunks_tbl, where=f"chunk_id = {_q(chunk_id)}", limit=1)
    elif doc_id and ordinal is not None:
        rows = _rows(chunks_tbl, where=f"doc_id = {_q(doc_id)} AND ordinal = {int(ordinal)}", limit=1)
    elif doc_id:
        rows = _rows(chunks_tbl, where=f"doc_id = {_q(doc_id)}", limit=1)
    else:
        return {"error": "需要 chunk_id 或 doc_id"}
    if not rows:
        return {"error": "没找到这个片段"}
    hit = rows[0]
    doc_id = hit["doc_id"]
    lo = max(0, int(hit["ordinal"]) - max(0, int(before)))
    hi = int(hit["ordinal"]) + max(0, int(after))
    window = _rows(chunks_tbl,
                   where=f"doc_id = {_q(doc_id)} AND ordinal >= {lo} AND ordinal <= {hi}")
    window = [w for w in window if lo <= int(w.get("ordinal", 0)) <= hi]
    window.sort(key=lambda w: int(w.get("ordinal", 0)))
    doc = (_rows(docs_tbl, where=f"doc_id = {_q(doc_id)}", limit=1) or [None])[0]
    return {
        "doc": doc,
        "focus": hit.get("chunk_id"),
        "chunks": [{
            "chunk_id": w.get("chunk_id"),
            "ordinal": w.get("ordinal"),
            "heading": w.get("heading") or "",
            "text": w.get("text") or "",
            "is_focus": w.get("chunk_id") == hit.get("chunk_id"),
        } for w in window],
    }


# --------------------------------------------------------------------------- #
# 文档管理
# --------------------------------------------------------------------------- #

def list_docs(limit: int = 100, order: str = "updated", filters: dict | None = None,
              query: str = "") -> dict:
    chunks_tbl, docs_tbl = ensure_tables()
    rows = _rows(docs_tbl)
    if filters and filters.get("doc_type"):
        types = filters["doc_type"]
        types = [types] if isinstance(types, str) else list(types)
        rows = [r for r in rows if r.get("doc_type") in types]
    if filters and filters.get("tags"):
        tags = filters["tags"]
        tags = [tags] if isinstance(tags, str) else list(tags)
        rows = [r for r in rows if any(t.strip() in (r.get("tags") or "") for t in tags if t)]
    if query:
        q = query.lower()
        rows = [r for r in rows
                if q in (r.get("doc_name") or "").lower() or q in (r.get("doc_path") or "").lower()]
    key = {"updated": lambda r: (r.get("updated_at") or ""),
           "added": lambda r: (r.get("added_at") or ""),
           "name": lambda r: (r.get("doc_name") or ""),
           "chunks": lambda r: int(r.get("n_chunks") or 0),
           "size": lambda r: int(r.get("n_chars") or 0)}.get(order, lambda r: r.get("updated_at") or "")
    rows.sort(key=key, reverse=order != "name")
    total = len(rows)
    return {"total": total, "returned": min(total, max(1, int(limit))),
            "items": rows[: max(1, int(limit))]}


def remove(doc_ids=None, path_prefix=None, pattern=None, all_docs=False, tags=None) -> dict:
    chunks_tbl, docs_tbl = ensure_tables()
    targets = []
    if all_docs:
        targets = _rows(docs_tbl)
    elif doc_ids:
        ids = [doc_ids] if isinstance(doc_ids, str) else list(doc_ids)
        for i in ids:
            targets.extend(_rows(docs_tbl, where=f"doc_id = {_q(i)}"))
    elif path_prefix or pattern:
        rows = _rows(docs_tbl)
        for r in rows:
            p = r.get("doc_path") or ""
            if path_prefix and p.startswith(os.path.abspath(os.path.expanduser(path_prefix))):
                targets.append(r)
            elif pattern and fnmatch.fnmatch(p, os.path.expanduser(pattern)):
                targets.append(r)
    elif tags:
        names = [tags] if isinstance(tags, str) else list(tags)
        for r in _rows(docs_tbl):
            if any(n.strip() in (r.get("tags") or "") for n in names if n):
                targets.append(r)
    else:
        return {"error": "没有指定删除目标（doc_ids / path_prefix / pattern / tags / all）"}

    seen, removed = set(), []
    for r in targets:
        did = r.get("doc_id")
        if not did or did in seen:
            continue
        seen.add(did)
        chunks_tbl.delete(f"doc_id = {_q(did)}")
        docs_tbl.delete(f"doc_id = {_q(did)}")
        removed.append({"doc_id": did, "name": r.get("doc_name"), "path": r.get("doc_path")})
    return {"removed": len(removed), "items": removed,
            "total_docs": docs_tbl.count_rows()}


def reindex(doc_ids=None) -> dict:
    """重建向量（换嵌入后端后必用）与全文索引。文本从原文件重读；笔记从库里取。"""
    emb = embedder()
    chunks_tbl, docs_tbl = ensure_tables()
    targets = _rows(docs_tbl, where=None)
    if doc_ids:
        ids = [doc_ids] if isinstance(doc_ids, str) else list(doc_ids)
        targets = [t for t in targets if t["doc_id"] in ids]
    cfg = load_config()
    size = int(cfg["chunk"].get("size") or DEFAULT_CHUNK)
    overlap = int(cfg["chunk"].get("overlap", DEFAULT_OVERLAP))
    report = {"docs": 0, "chunks": 0, "failed": [], "embedder": emb.describe()}
    for doc in targets:
        try:
            if doc.get("source") == "note":
                old = _rows(chunks_tbl, where=f"doc_id = {_q(doc['doc_id'])}")
                old.sort(key=lambda r: int(r.get("ordinal") or 0))
                pieces = [{"heading": r.get("heading") or "", "text": r.get("text") or ""}
                          for r in old]
            else:
                path = doc.get("doc_path")
                if not path or not os.path.exists(path):
                    report["failed"].append({"doc_id": doc["doc_id"], "error": "源文件已不存在"})
                    continue
                blocks = loaders.read_blocks(path, doc.get("doc_type"))
                pieces = loaders.chunk_blocks(blocks, size, overlap)
            if not pieces:
                continue
            _write_doc(chunks_tbl, docs_tbl, doc, pieces, emb)
            report["docs"] += 1
            report["chunks"] += len(pieces)
        except Exception as exc:
            report["failed"].append({"doc_id": doc["doc_id"], "error": f"{type(exc).__name__}: {exc}"})
    report["fts"] = _ensure_fts(chunks_tbl)
    return report


def sync() -> dict:
    """重扫登记过的来源目录：新增/变更入库，源文件消失的从库里删掉。"""
    cfg = load_config()
    sources = cfg.get("sources") or []
    if not sources:
        return {"message": "还没有登记过来源目录。先 kb_add 一个目录，之后 sync 才能增量重扫。",
                "sources": []}
    chunks_tbl, docs_tbl = ensure_tables()
    report = {"sources": [], "added": 0, "updated": 0, "skipped": 0, "failed": 0,
              "pruned": [], "chunks": 0}
    alive = set()
    for src in sources:
        path = src.get("path")
        if not path or not os.path.exists(path):
            report["sources"].append({"path": path, "status": "missing"})
            continue
        if os.path.isdir(path):
            for ap in loaders.iter_files([path], recursive=True):
                alive.add(ap)
        else:
            alive.add(os.path.abspath(path))
        r = add_paths([path], tags=src.get("tags"), record_source=False, build_fts=False)
        report["added"] += len(r["added"])
        report["updated"] += len(r["updated"])
        report["skipped"] += len(r["skipped"])
        report["failed"] += len(r["failed"])
        report["chunks"] += r["chunks"]
        report["sources"].append({"path": path, "status": "ok",
                                  "added": len(r["added"]), "updated": len(r["updated"]),
                                  "skipped": len(r["skipped"]), "failed": len(r["failed"])})

    # 只清理「文件来源」里已经不存在的，笔记（note://）不动
    for doc in _rows(docs_tbl):
        if doc.get("source") != "file":
            continue
        p = doc.get("doc_path")
        if p and p not in alive and any(p.startswith(os.path.abspath(str(s.get("path") or "/__none__")))
                                        for s in sources):
            chunks_tbl.delete(f"doc_id = {_q(doc['doc_id'])}")
            docs_tbl.delete(f"doc_id = {_q(doc['doc_id'])}")
            report["pruned"].append({"doc_id": doc["doc_id"], "path": p})
    if report["added"] or report["updated"] or report["pruned"]:
        _ensure_fts(chunks_tbl, rebuild=True)
    report["total_docs"] = docs_tbl.count_rows()
    return report


# --------------------------------------------------------------------------- #
# 统计与看板
# --------------------------------------------------------------------------- #

def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def stats(include_backends: bool = True) -> dict:
    emb = embedder()
    chunks_tbl, docs_tbl = ensure_tables()
    docs = _rows(docs_tbl)
    total_chunks = chunks_tbl.count_rows()
    by_type: dict[str, dict] = {}
    by_tag: dict[str, int] = {}
    for d in docs:
        t = d.get("doc_type") or "other"
        slot = by_type.setdefault(t, {"type": t, "docs": 0, "chunks": 0, "chars": 0})
        slot["docs"] += 1
        slot["chunks"] += int(d.get("n_chunks") or 0)
        slot["chars"] += int(d.get("n_chars") or 0)
        for tag in (d.get("tags") or "").split(","):
            tag = tag.strip()
            if tag:
                by_tag[tag] = by_tag.get(tag, 0) + 1
    newest = max((d.get("updated_at") or "" for d in docs), default="")
    oldest = min((d.get("added_at") or "" for d in docs), default="")
    fts = _STATE.get("fts")
    if fts is None:
        fts = _ensure_fts(chunks_tbl, rebuild=False) if total_chunks else ""
    info = {
        "kb_home": KB_HOME,
        "lance_dir": LANCE_DIR,
        "db_bytes": _dir_size(LANCE_DIR) if os.path.isdir(LANCE_DIR) else 0,
        "docs": len(docs),
        "chunks": total_chunks,
        "chars": sum(int(d.get("n_chars") or 0) for d in docs),
        "avg_chunk_chars": round(sum(int(d.get("n_chars") or 0) for d in docs) / total_chunks, 1)
        if total_chunks else 0,
        "by_type": sorted(by_type.values(), key=lambda x: -x["docs"]),
        "top_tags": sorted(({"tag": k, "docs": v} for k, v in by_tag.items()),
                           key=lambda x: -x["docs"])[:20],
        "chunk_size": load_config()["chunk"].get("size"),
        "overlap": load_config()["chunk"].get("overlap"),
        "embedder": emb.info(),
        "embedder_label": emb.describe(),
        "fts_tokenizer": fts or "未建立",
        "newest_doc": newest,
        "oldest_doc": oldest,
        "sources": load_config().get("sources") or [],
        "schema_version": SCHEMA_VERSION,
    }
    if include_backends:
        info["backends"] = embedders.available_backends()
    if emb.name == "hash":
        info["caveat"] = ("当前用的是兜底哈希向量（没有语义泛化能力），检索质量主要靠全文检索那一路。"
                          "想要更好的语义召回：本机起 ollama 拉 bge-m3，或安装 fastembed，然后 kb_reindex。")
    return info


def set_chunk_params(size=None, overlap=None) -> dict:
    cfg = load_config()
    if size:
        cfg["chunk"]["size"] = max(200, min(int(size), 4000))
    if overlap is not None:
        cfg["chunk"]["overlap"] = max(0, min(int(overlap), cfg["chunk"]["size"] // 2))
    save_config(cfg)
    return {"chunk": cfg["chunk"], "note": "已有文档的切块不变，要对新参数生效请 kb_reindex"}


def dashboard_payload(max_docs: int = 800, max_chunks: int = 1500) -> dict:
    emb = embedder()
    chunks_tbl, docs_tbl = ensure_tables()
    info = stats(include_backends=False)
    docs = _rows(docs_tbl)
    docs.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
    rows = [{
        "doc_id": d.get("doc_id"),
        "name": d.get("doc_name"),
        "path": d.get("doc_path"),
        "type": d.get("doc_type"),
        "chunks": int(d.get("n_chunks") or 0),
        "chars": int(d.get("n_chars") or 0),
        "tags": d.get("tags") or "",
        "updated_at": d.get("updated_at") or "",
        "source": d.get("source") or "file",
    } for d in docs[:max_docs]]
    sample = _rows(chunks_tbl, columns=["doc_id", "doc_name", "doc_type", "heading", "text",
                                        "ordinal", "n_chars"], limit=max_chunks)
    chunks = [{
        "doc_id": c.get("doc_id"),
        "doc_name": c.get("doc_name"),
        "doc_type": c.get("doc_type"),
        "heading": c.get("heading") or "",
        "text": (c.get("text") or "")[:600],
        "ordinal": c.get("ordinal"),
        "n_chars": c.get("n_chars"),
    } for c in sample]
    return {
        "meta": {
            "kb_home": KB_HOME,
            "generated_at": now_iso(),
            "docs_in_payload": len(rows),
            "chunks_in_payload": len(chunks),
            "docs_total": info["docs"],
            "chunks_total": info["chunks"],
            "embedder": info["embedder_label"],
            "fts": info["fts_tokenizer"],
            "db_bytes": info["db_bytes"],
            "caveat": info.get("caveat") or "",
        },
        "stats": info,
        "docs": rows,
        "chunks": chunks,
    }


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #

def _print(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kb.py", description="local knowledge base (LanceDB)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add", help="入库：文件或目录")
    p.add_argument("paths", nargs="+")
    p.add_argument("--tags", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--overlap", type=int, default=None)

    p = sub.add_parser("note", help="入库一段文字")
    p.add_argument("--name", required=True)
    p.add_argument("--text", default=None)
    p.add_argument("--text-file", default=None)
    p.add_argument("--tags", default=None)

    p = sub.add_parser("search")
    p.add_argument("-q", "--query", required=True)
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "keyword"])
    p.add_argument("--doc-type", default=None)
    p.add_argument("--path-prefix", default=None)
    p.add_argument("--tags", default=None)

    p = sub.add_parser("get")
    p.add_argument("--chunk-id", default=None)
    p.add_argument("--doc-id", default=None)
    p.add_argument("--before", type=int, default=1)
    p.add_argument("--after", type=int, default=2)

    p = sub.add_parser("docs")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--order", default="updated")

    sub.add_parser("stats")

    p = sub.add_parser("rm")
    p.add_argument("--doc-id", default=None)
    p.add_argument("--path-prefix", default=None)
    p.add_argument("--pattern", default=None)
    p.add_argument("--tags", default=None)
    p.add_argument("--all", action="store_true")

    sub.add_parser("sync")

    p = sub.add_parser("reindex")
    p.add_argument("--doc-id", default=None)

    sub.add_parser("export-dev")
    p = sub.add_parser("chunk-config")
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--overlap", type=int, default=None)

    args = ap.parse_args(argv)
    cmd = args.cmd

    if cmd == "add":
        r = add_paths(args.paths, tags=args.tags, force=args.force,
                      recursive=not args.no_recursive, chunk_size=args.chunk_size,
                      overlap=args.overlap, log=lambda m: print(m, file=sys.stderr))
        _print(r)
        return 0
    if cmd == "note":
        text = args.text
        if args.text_file:
            with open(args.text_file, encoding="utf-8") as fh:
                text = fh.read()
        if not text:
            print("需要 --text 或 --text-file", file=sys.stderr)
            return 2
        _print(add_note(args.name, text, tags=args.tags))
        return 0
    if cmd == "search":
        _print(search(args.query, args.k, args.mode, {
            "doc_type": args.doc_type, "path_prefix": args.path_prefix, "tags": args.tags}))
        return 0
    if cmd == "get":
        _print(get_chunk(args.chunk_id, args.doc_id, args.before, args.after))
        return 0
    if cmd == "docs":
        _print(list_docs(args.limit, args.order))
        return 0
    if cmd == "stats":
        _print(stats())
        return 0
    if cmd == "rm":
        _print(remove(doc_ids=args.doc_id, path_prefix=args.path_prefix,
                      pattern=args.pattern, tags=args.tags, all_docs=args.all))
        return 0
    if cmd == "sync":
        _print(sync())
        return 0
    if cmd == "reindex":
        _print(reindex(args.doc_id))
        return 0
    if cmd == "chunk-config":
        _print(set_chunk_params(args.size, args.overlap))
        return 0
    if cmd == "export-dev":
        out = os.path.join(PLUGIN_ROOT, "dashboard", "data.dev.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(dashboard_payload(), fh, ensure_ascii=False)
        _print({"written": out})
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
