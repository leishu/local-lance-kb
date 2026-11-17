#!/usr/bin/env python3
"""lance-kb MCP 服务端（stdio，协议自己实现，不依赖 mcp SDK）。

数据在 LanceDB（data/lancedb），由 kb.py 负责入库与检索。
工具：
  kb_stats        库的规模、类型分布、嵌入后端、索引状态
  kb_add          把文件/目录/一段文字加进库里（增量：按内容哈希跳过没变的）
  kb_search       混合检索（向量 + 全文 RRF），返回带出处的片段
  kb_open         按 chunk_id / doc_id 取上下文，看命中片段前后的原文
  kb_docs         文档清单（过滤、排序）
  kb_remove       删文档（按 id / 路径前缀 / glob / 标签 / 全清）
  kb_reindex      重建向量与全文索引（换嵌入后端后必用）
  kb_sync         重扫登记过的来源目录，增量更新并清理已删除文件
  open_dashboard  打开知识库看板（MCP App）
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN_ROOT = os.path.dirname(HERE)
DASHBOARD_HTML = os.path.join(PLUGIN_ROOT, "dashboard", "index.html")
DASHBOARD_URI = "ui://lance-kb/dashboard"
PAYLOAD_PLACEHOLDER = "__LANCE_KB_DATA_B64__"


def _plugin_version() -> str:
    try:
        with open(os.path.join(PLUGIN_ROOT, ".codebuddy-plugin", "plugin.json"),
                  encoding="utf-8") as fh:
            return str(json.load(fh).get("version") or "0.0.0")
    except (OSError, ValueError):
        return "0.0.0"


VERSION = _plugin_version()


def log(msg: str) -> None:
    sys.stderr.write(f"[lance-kb] {msg}\n")
    sys.stderr.flush()


def text_of(obj) -> str:
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _fmt_chars(n: int) -> str:
    n = int(n or 0)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M 字"
    if n >= 1000:
        return f"{n / 1000:.1f}k 字"
    return f"{n} 字"


def _cite(hit: dict) -> str:
    """引用写法统一成「文档名 › 小节」，模型好抄，用户点得回原文。"""
    name = hit.get("doc_name") or "?"
    head = hit.get("heading") or ""
    return f"{name} › {head}" if head else name


# --------------------------------------------------------------------------- #
# 工具实现
# --------------------------------------------------------------------------- #

def tool_add(args: dict) -> dict:
    paths = args.get("paths") or []
    if isinstance(paths, str):
        paths = [p for p in paths.replace("\n", ",").split(",") if p.strip()]
    name, text = args.get("name"), args.get("text")
    if text and not paths:
        return kb.add_note(name or "随手记", text, tags=args.get("tags"),
                           chunk_size=args.get("chunk_size"), overlap=args.get("overlap"))
    if not paths:
        return {"error": "需要 paths（文件/目录/通配）或 text（一段文字）"}
    return kb.add_paths(paths, tags=args.get("tags"), force=bool(args.get("force")),
                        recursive=bool(args.get("recursive", True)),
                        chunk_size=args.get("chunk_size"), overlap=args.get("overlap"),
                        log=log)


def _filters(args: dict) -> dict:
    return {
        "doc_type": args.get("doc_type") or args.get("type"),
        "doc_id": args.get("doc_id"),
        "path_prefix": args.get("path_prefix"),
        "path_contains": args.get("path_contains"),
        "tags": args.get("tags"),
    }


TOOLS = [
    {
        "name": "kb_stats",
        "description": "本地知识库概况：文档数、片段数、总字数、按类型分布、标签、"
                       "当前嵌入后端与维度、全文索引分词器、库目录与磁盘占用、已登记的来源目录。"
                       "回答任何与知识库有关的问题前先调它，确认库里有什么、能不能语义检索。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "kb_add",
        "description": "把资料加进本地知识库。两种用法：① paths 传文件/目录/通配（支持 "
                       "md/txt/pdf/docx/html/csv/json/ipynb/常见代码文件，目录递归）；"
                       "② name + text 直接存一段文字（会议记录、结论、口述笔记）。"
                       "按内容哈希增量：文件没改就跳过，改了只重建那一个文档。"
                       "tags 便于以后按标签过滤（如 项目A,规范）。第一次用先问用户要索引哪个目录。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"},
                          "description": "文件、目录或通配路径，如 [\"/home/me/docs\"] 或 [\"~/notes/*.md\"]"},
                "text": {"type": "string", "description": "直接入库的一段文字（与 paths 二选一）"},
                "name": {"type": "string", "description": "text 模式的标题"},
                "tags": {"type": "string", "description": "逗号分隔的标签，如 \"项目A,接口文档\""},
                "force": {"type": "boolean", "default": False,
                          "description": "即使内容哈希没变也重建该文档"},
                "recursive": {"type": "boolean", "default": True},
                "chunk_size": {"type": "integer", "description": "片段目标字数，默认 700"},
                "overlap": {"type": "integer", "description": "片段重叠字数，默认 120"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "kb_search",
        "description": "检索本地知识库，返回带出处的片段。mode=hybrid（默认，向量+全文 RRF 融合）"
                       "最稳；纯专有名词/代码符号用 keyword；问概念、换种说法找同类内容用 vector。"
                       "回答用户问题前先检索，再基于返回的片段作答，并按「文档名 › 小节」标出处；"
                       "片段被截断（truncated=true）时用 kb_open 取全文。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索词或自然语言问题"},
                "k": {"type": "integer", "default": 8, "description": "返回片段数（1-50）"},
                "mode": {"type": "string", "enum": ["hybrid", "vector", "keyword"],
                         "default": "hybrid"},
                "doc_type": {"type": "string",
                             "description": "只看某类文档：md/txt/pdf/docx/html/csv/json/note/code"},
                "tags": {"type": "string", "description": "按标签过滤（逗号分隔，命中任一）"},
                "path_prefix": {"type": "string", "description": "只看某目录下的文档"},
                "path_contains": {"type": "string", "description": "路径包含某段字符"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "kb_open",
        "description": "取一个命中片段的上下文：给 chunk_id（kb_search 返回的）连同前后各 N 个片段，"
                       "或给 doc_id 看该文档开头。用于回答需要完整段落/代码块、或片段被截断时。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string"},
                "doc_id": {"type": "string"},
                "ordinal": {"type": "integer", "description": "配合 doc_id 定位具体片段序号"},
                "before": {"type": "integer", "default": 1},
                "after": {"type": "integer", "default": 2},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "kb_docs",
        "description": "列出知识库里的文档（名称、路径、类型、片段数、字数、标签、更新时间），"
                       "可按 query/doc_type/tags 过滤，order=updated|added|name|chunks|size。"
                       "用户问「库里有哪些东西」「某目录收进来了吗」时用。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "按文件名/路径片段过滤"},
                "doc_type": {"type": "string"},
                "tags": {"type": "string"},
                "limit": {"type": "integer", "default": 50},
                "order": {"type": "string",
                          "enum": ["updated", "added", "name", "chunks", "size"],
                          "default": "updated"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "kb_remove",
        "description": "从知识库里删文档（同时删其片段）。doc_ids 精确删；path_prefix 删某目录下全部；"
                       "pattern 用 glob（如 \"*/tmp/*.md\"）；tags 按标签删；all=true 清空重建。"
                       "删除不可撤销，调用前先跟用户确认范围。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_ids": {"type": "array", "items": {"type": "string"}},
                "path_prefix": {"type": "string"},
                "pattern": {"type": "string"},
                "tags": {"type": "string"},
                "all": {"type": "boolean", "default": False},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "kb_reindex",
        "description": "重建向量与全文索引。换嵌入后端（例如从兜底 hash 换成 ollama/bge-m3）、"
                       "改过切块参数、或检索结果明显不对时用。doc_ids 只重建指定文档；不传则全量。"
                       "全量重建会按当前后端重算所有向量，文档多时较慢。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_ids": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "kb_sync",
        "description": "重扫之前 kb_add 登记过的目录：新文件入库、改过的重建、源文件已删除的从库里清掉。"
                       "适合定期保鲜（配合定时任务），比每次全量 kb_add 快得多。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "open_dashboard",
        "description": "在对话下方打开本地知识库看板：规模 KPI、类型分布、标签、文档清单与片段浏览，"
                       "支持按类型/标签/关键词筛选，点文档看它的片段。用户想看「库里都有什么」"
                       "或说「打开看板」时调用。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "focus": {"type": "string", "enum": ["overview", "docs", "chunks"],
                          "default": "overview"},
            },
        },
        "_meta": {"ui": {"resourceUri": DASHBOARD_URI}},
    },
]


def call_tool(name: str, args: dict) -> dict:
    args = args or {}

    if name == "kb_stats":
        data = kb.stats()
        head = (f"{data['docs']} 个文档、{data['chunks']} 个片段、{_fmt_chars(data['chars'])}"
                f"（平均每片段 {data['avg_chunk_chars']} 字）。")
        emb = data["embedder_label"]
        idx = f"\n嵌入后端：{emb}；全文索引分词器：{data['fts_tokenizer']}。"
        tail = ""
        if data.get("caveat"):
            tail = "\n注意：" + data["caveat"]
        if not data["docs"]:
            tail += ("\n库里还是空的：让用户指定一个目录，或直接用 kb_add 加文件/一段文字。")
        types = "、".join(f"{t['type']} {t['docs']}" for t in data["by_type"][:8])
        return {"content": [{"type": "text",
                             "text": head + (f"\n类型分布：{types}。" if types else "")
                                     + idx + tail + "\n\n" + text_of(data)}],
                "structuredContent": data}

    if name == "kb_add":
        data = tool_add(args)
        if data.get("error"):
            return {"content": [{"type": "text", "text": data["error"]}],
                    "structuredContent": data, "isError": True}
        if "doc_id" in data:  # note 模式
            return {"content": [{"type": "text",
                                 "text": f"已存入「{data['name']}」：{data['chunks']} 个片段，"
                                         f"{_fmt_chars(data['chars'])}（{data['embedder']}）。"}],
                    "structuredContent": data}
        head = (f"入库完成（{data['elapsed_sec']}s，{data['embedder']}）："
                f"新增 {len(data['added'])}，更新 {len(data['updated'])}，"
                f"跳过 {len(data['skipped'])}，失败 {len(data['failed'])}；"
                f"本次写入 {data['chunks']} 个片段，库里现有 {data['total_docs']} 个文档。")
        lines = []
        for item in data["added"][:20]:
            lines.append(f"+ {item['name']}\t{item['chunks']} 块\t{item['path']}")
        for item in data["updated"][:20]:
            lines.append(f"~ {item['name']}\t{item['chunks']} 块\t{item['path']}")
        for item in data["failed"][:10]:
            lines.append(f"× {item['path']}\t{item['error']}")
        more = len(data["added"]) + len(data["updated"]) > 40
        return {"content": [{"type": "text",
                             "text": head + ("\n" + "\n".join(lines) if lines else "")
                                     + ("\n（列表已截断，完整清单见结构化结果）" if more else "")}],
                "structuredContent": data}

    if name == "kb_search":
        data = kb.search(args.get("query") or "", int(args.get("k", 8)),
                         args.get("mode", "hybrid"), _filters(args))
        if data.get("error"):
            return {"content": [{"type": "text", "text": data["error"]}],
                    "structuredContent": data, "isError": True}
        if data.get("empty"):
            return {"content": [{"type": "text", "text": data["message"]}],
                    "structuredContent": data}
        if not data["hits"]:
            return {"content": [{"type": "text",
                                 "text": f"「{data['query']}」没检索到相关片段"
                                         f"（mode={data['mode']}）。可以换更宽泛的说法，"
                                         f"或用 mode=keyword 试专有名词。"}],
                    "structuredContent": data}
        lines = []
        for n, hit in enumerate(data["hits"], 1):
            marks = []
            if hit["ranks"].get("vector"):
                marks.append(f"向量#{hit['ranks']['vector']}")
            if hit["ranks"].get("keyword"):
                marks.append(f"全文#{hit['ranks']['keyword']}")
            meta = "，".join(marks) or "-"
            lines.append(f"[{n}] {_cite(hit)}（{meta}，{hit['n_chars']} 字）\n{hit['text']}")
        head = (f"检索「{data['query']}」：{len(data['hits'])} 个片段"
                f"（mode={data['mode']}，{data['embedder']}）。引用时写「文档名 › 小节」。")
        return {"content": [{"type": "text", "text": head + "\n\n" + "\n\n".join(lines)}],
                "structuredContent": data}

    if name == "kb_open":
        data = kb.get_chunk(args.get("chunk_id"), args.get("doc_id"),
                            int(args.get("before", 1)), int(args.get("after", 2)),
                            args.get("ordinal"))
        if data.get("error"):
            return {"content": [{"type": "text", "text": data["error"]}],
                    "structuredContent": data, "isError": True}
        doc = data.get("doc") or {}
        head = f"{doc.get('doc_name') or '?'}（{doc.get('doc_type')}）"
        if doc.get("doc_path"):
            head += f"\n{doc['doc_path']}"
        body = []
        for c in data["chunks"]:
            mark = "▶" if c["is_focus"] else " "
            title = f" [{c['heading']}]" if c["heading"] else ""
            body.append(f"{mark} #{c['ordinal']}{title}\n{c['text']}")
        return {"content": [{"type": "text", "text": head + "\n\n" + "\n\n".join(body)}],
                "structuredContent": data}

    if name == "kb_docs":
        data = kb.list_docs(int(args.get("limit", 50)), args.get("order", "updated"),
                            _filters(args), args.get("query") or "")
        lines = [f"{d['doc_name']}\t{d.get('doc_type')}\t{d.get('n_chunks')} 块\t"
                 f"{_fmt_chars(d.get('n_chars'))}\t{d.get('tags') or '-'}\t{d.get('updated_at', '')[:10]}"
                 for d in data["items"]]
        head = f"共 {data['total']} 个文档"
        if data["returned"] < data["total"]:
            head += f"（列出前 {data['returned']} 个）"
        return {"content": [{"type": "text", "text": head + "\n" + "\n".join(lines)}],
                "structuredContent": data}

    if name == "kb_remove":
        data = kb.remove(doc_ids=args.get("doc_ids"), path_prefix=args.get("path_prefix"),
                         pattern=args.get("pattern"), tags=args.get("tags"),
                         all_docs=bool(args.get("all")))
        if data.get("error"):
            return {"content": [{"type": "text", "text": data["error"]}],
                    "structuredContent": data, "isError": True}
        lines = [f"- {i['name']}\t{i['path']}" for i in data["items"][:20]]
        return {"content": [{"type": "text",
                             "text": f"已删除 {data['removed']} 个文档，库里剩 "
                                     f"{data['total_docs']} 个。"
                                     + ("\n" + "\n".join(lines) if lines else "")}],
                "structuredContent": data}

    if name == "kb_reindex":
        data = kb.reindex(args.get("doc_ids"))
        errs = data.get("failed") or []
        head = (f"重建完成（{data['embedder']}）：{data['docs']} 个文档、{data['chunks']} 个片段，"
                f"全文索引分词器 {data.get('fts') or '未建立'}。")
        if errs:
            head += f"\n{len(errs)} 个文档失败：" + "；".join(
                f"{e.get('doc_id')} {e.get('error')}" for e in errs[:3])
        return {"content": [{"type": "text", "text": head}], "structuredContent": data}

    if name == "kb_sync":
        data = kb.sync()
        if data.get("message"):
            return {"content": [{"type": "text", "text": data["message"]}],
                    "structuredContent": data}
        head = (f"同步完成：新增 {data['added']}，更新 {data['updated']}，"
                f"跳过 {data['skipped']}，失败 {data['failed']}，"
                f"清理已删除 {len(data['pruned'])}；库里现有 {data['total_docs']} 个文档。")
        miss = [s["path"] for s in data["sources"] if s.get("status") == "missing"]
        if miss:
            head += "\n以下来源目录已不存在：" + "、".join(miss[:5])
        return {"content": [{"type": "text", "text": head}], "structuredContent": data}

    if name == "open_dashboard":
        data = kb.dashboard_payload()
        m = data["meta"]
        if not m["docs_total"]:
            msg = "看板已打开，但库里还没有任何文档 —— 先 kb_add 收录一个目录或一段文字。"
        else:
            msg = (f"已打开知识库看板：{m['docs_total']} 个文档、{m['chunks_total']} 个片段"
                   f"（{m['embedder']}）。可按类型/标签/关键词筛选，点文档看它的片段。")
        return {
            "content": [{"type": "text", "text": msg}],
            "structuredContent": {"opened": True, "docs": m["docs_total"],
                                  "focus": args.get("focus", "overview")},
            "_meta": {"ui": {"resourceUri": _payload_uri(data)}},
        }

    raise ValueError(f"未知工具: {name}")


# --------------------------------------------------------------------------- #
# 看板资源
# --------------------------------------------------------------------------- #

def _payload_bytes(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"),
                      sort_keys=True).encode("utf-8")


def _payload_uri(data: dict) -> str:
    ver = hashlib.sha1(_payload_bytes(data)).hexdigest()[:12]
    return f"{DASHBOARD_URI}?v={ver}"


def render_dashboard() -> str:
    if not os.path.exists(DASHBOARD_HTML):
        raise RuntimeError(f"看板文件缺失：{DASHBOARD_HTML}")
    with open(DASHBOARD_HTML, encoding="utf-8") as fh:
        tpl = fh.read()
    if PAYLOAD_PLACEHOLDER not in tpl:
        raise RuntimeError(f"看板模板缺少 {PAYLOAD_PLACEHOLDER} 占位符")
    payload = kb.dashboard_payload()
    blob = base64.b64encode(gzip.compress(_payload_bytes(payload), 9)).decode("ascii")
    return tpl.replace(PAYLOAD_PLACEHOLDER, blob)


# --------------------------------------------------------------------------- #
# 协议循环
# --------------------------------------------------------------------------- #

def handle(req: dict) -> dict | None:
    method = req.get("method")
    rid = req.get("id")
    params = req.get("params") or {}

    def ok(result):
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def err(code, message):
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}

    if method == "initialize":
        return ok({
            "protocolVersion": params.get("protocolVersion") or "2024-11-05",
            "capabilities": {"tools": {"listChanged": False},
                             "resources": {"listChanged": False, "subscribe": False}},
            "serverInfo": {"name": "lance-kb", "version": VERSION},
            "instructions": "本地知识库（LanceDB）。资料在本地，不联网、不出机器。"
                            "回答与用户资料有关的问题前，先 kb_search 检索，基于片段作答并标出处"
                            "（文档名 › 小节）；片段被截断就用 kb_open 取全文。"
                            "用户要「记住/收录/整理」某目录或某段内容时用 kb_add；"
                            "问库里有什么先 kb_stats；想看图表用 open_dashboard。",
        })
    if method in ("notifications/initialized", "notifications/cancelled"):
        return None
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": TOOLS})
    if method == "tools/call":
        try:
            return ok(call_tool(params.get("name"), dict(params.get("arguments") or {})))
        except kb.NeedReindex as exc:
            return ok({"content": [{"type": "text", "text": str(exc)}],
                       "isError": True})
        except Exception as exc:
            log(traceback.format_exc())
            return ok({"content": [{"type": "text", "text": f"工具执行失败: {exc}"}],
                       "isError": True})
    if method == "resources/list":
        return ok({"resources": [{
            "uri": DASHBOARD_URI,
            "name": "本地知识库看板",
            "description": "文档/片段规模、类型与标签分布、文档清单浏览",
            "mimeType": "text/html;profile=mcp-app",
        }]})
    if method == "resources/read":
        uri = str(params.get("uri") or "")
        if uri.startswith(DASHBOARD_URI):
            return ok({"contents": [{
                "uri": uri,
                "mimeType": "text/html;profile=mcp-app",
                "text": render_dashboard(),
            }]})
        return err(-32002, f"未知资源: {uri}")
    if method == "prompts/list":
        return ok({"prompts": []})
    return err(-32601, f"不支持的方法: {method}")


def main() -> int:
    log(f"start, kb_home={kb.KB_HOME}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            resp = handle(req)
        except Exception as exc:
            log(traceback.format_exc())
            resp = {"jsonrpc": "2.0", "id": req.get("id"),
                    "error": {"code": -32603, "message": str(exc)}}
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
