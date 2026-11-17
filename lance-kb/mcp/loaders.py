#!/usr/bin/env python3
"""文件解析与切块：把各种格式读成「带标题的段落」，再切成检索片段。

分两层是为了让切块策略只写一遍：
    read_blocks()  → [(标题路径, 正文)]   各格式自己的解析逻辑
    chunk_blocks() → [片段]               通用打包逻辑（目标长度 + 重叠）

PDF/DOCX 依赖第三方库，没装时抛 MissingDep，由上层跳过一个文件而不是整批失败。
"""

from __future__ import annotations

import html
import io
import json
import os
import re

# 扩展名 → 类型。类型只用于展示与过滤，切块时再分「结构化 / 代码 / 表格」三路。
TYPE_BY_EXT = {
    ".md": "md", ".markdown": "md", ".mdx": "md",
    ".txt": "txt", ".text": "txt", ".log": "txt", ".rst": "txt",
    ".pdf": "pdf",
    ".docx": "docx",
    ".html": "html", ".htm": "html",
    ".csv": "csv", ".tsv": "csv",
    ".json": "json", ".jsonl": "json", ".yaml": "json", ".yml": "json",
    ".toml": "json", ".ini": "json", ".conf": "json",
    ".ipynb": "notebook",
    ".py": "code", ".js": "code", ".ts": "code", ".tsx": "code", ".jsx": "code",
    ".vue": "code", ".java": "code", ".go": "code", ".rs": "code", ".c": "code",
    ".h": "code", ".cpp": "code", ".hpp": "code", ".cs": "code", ".rb": "code",
    ".php": "code", ".kt": "code", ".swift": "code", ".scala": "code",
    ".sql": "code", ".sh": "code", ".bash": "code", ".zsh": "code",
    ".css": "code", ".scss": "code", ".less": "code", ".lua": "code",
    ".r": "code", ".pl": "code", ".m": "code", ".dart": "code",
    ".xml": "code", ".gradle": "code", ".proto": "code",
}

# 明显是二进制/无意义的目录，递归扫描时直接跳过
SKIP_DIRS = {
    ".git", ".svn", ".hg", "node_modules", "__pycache__", ".venv", "venv",
    ".idea", ".vscode", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", "target", ".next", ".nuxt", ".cache", ".Trash-0",
    ".workbuddy", "site-packages", ".tox", ".gradle", ".terraform",
}

SKIP_FILES = {
    ".DS_Store", "Thumbs.db", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Cargo.lock", "go.sum", "uv.lock",
}

MAX_FILE_BYTES = 8 * 1024 * 1024


class MissingDep(RuntimeError):
    """缺可选依赖（pypdf / python-docx）。"""


def detect_type(path: str) -> str | None:
    ext = os.path.splitext(path)[1].lower()
    if ext in TYPE_BY_EXT:
        return TYPE_BY_EXT[ext]
    # 无扩展名的常见文本文件（LICENSE、Makefile、Dockerfile…）
    base = os.path.basename(path)
    if "." not in base and base[:1].isupper():
        return "txt"
    return None


def is_candidate(path: str) -> bool:
    if os.path.basename(path) in SKIP_FILES:
        return False
    return detect_type(path) is not None


def iter_files(paths, recursive: bool = True):
    """把用户给的文件/目录/通配展开成文件列表，顺带跳过垃圾目录。"""
    import glob as _glob

    seen = set()
    for raw in paths:
        item = os.path.expanduser(str(raw).strip())
        if not item:
            continue
        matches = _glob.glob(item, recursive=True) if any(c in item for c in "*?[") else [item]
        for p in matches:
            if os.path.isfile(p):
                ap = os.path.abspath(p)
                if ap not in seen and is_candidate(ap):
                    seen.add(ap)
                    yield ap
            elif os.path.isdir(p):
                for root, dirs, files in os.walk(p):
                    dirs[:] = sorted(d for d in dirs
                                     if d not in SKIP_DIRS and not d.startswith("."))
                    for name in sorted(files):
                        ap = os.path.abspath(os.path.join(root, name))
                        if ap in seen or not is_candidate(ap):
                            continue
                        try:
                            if os.path.getsize(ap) > MAX_FILE_BYTES:
                                continue
                        except OSError:
                            continue
                        seen.add(ap)
                        yield ap
                    if not recursive:
                        break


# --------------------------------------------------------------------------- #
# 各格式 → 段落块
# --------------------------------------------------------------------------- #

def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def blocks_markdown(text: str):
    """按标题层级切成段，标题路径拼成「一级 > 二级」，片段带上它才好被检索到。"""
    lines = text.split("\n")
    out = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_fence = False

    def path() -> str:
        return " > ".join(t for _, t in stack)

    def flush():
        body = _clean("\n".join(buf))
        if body:
            out.append((path(), body))
        buf.clear()

    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            buf.append(line)
            continue
        m = None if in_fence else MD_HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip().strip("#").strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            continue
        buf.append(line)
    flush()
    return out


def blocks_text(text: str):
    """普通文本：空行分段，段落本身当作唯一结构。"""
    paras = [p for p in re.split(r"\n\s*\n", _clean(text)) if p.strip()]
    return [("", p.strip()) for p in paras]


CODE_HEAD = re.compile(
    r"^(?:def |class |async def |function |export |public |private |protected |"
    r"impl |func |fn |struct |interface |type |CREATE |SELECT |INSERT |UPDATE )")


def blocks_code(text: str, max_lines: int = 80):
    """代码：优先在顶层 def/class 处断开，找不到就按行数硬切。"""
    lines = text.split("\n")
    out, buf = [], []
    for line in lines:
        starts = bool(CODE_HEAD.match(line)) and not line.startswith((" ", "\t"))
        if starts and len(buf) >= 12:
            body = _clean("\n".join(buf))
            if body:
                out.append(("", body))
            buf = []
        buf.append(line)
        if len(buf) >= max_lines:
            body = _clean("\n".join(buf))
            if body:
                out.append(("", body))
            buf = buf[-12:]  # 留一点上下文，避免跨块的函数被切断得太干净
    body = _clean("\n".join(buf))
    if body:
        out.append(("", body))
    return out


def blocks_csv(path: str, text: str, rows_per_chunk: int = 40):
    """表格：每个片段都重复表头，否则后半截数据没有列名可用。"""
    import csv as _csv

    try:
        dialect = _csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except Exception:
        dialect = _csv.excel
    reader = _csv.reader(io.StringIO(text), dialect)
    rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        return []
    header = rows[0]
    out, batch = [], []
    for r in rows[1:]:
        batch.append(r)
        if len(batch) >= rows_per_chunk:
            out.append(("行 %d-%d" % (len(out) * rows_per_chunk + 1,
                                     len(out) * rows_per_chunk + len(batch)),
                        _csv_text(header, batch)))
            batch = []
    if batch:
        out.append(("行 %d-%d" % (len(out) * rows_per_chunk + 1,
                                 len(out) * rows_per_chunk + len(batch)),
                    _csv_text(header, batch)))
    return out or [("", _csv_text(header, []))]


def _csv_text(header, rows) -> str:
    lines = [" | ".join(c.strip() for c in header)]
    for r in rows:
        lines.append(" | ".join(c.strip() for c in r))
    return "\n".join(lines)


def blocks_html(text: str):
    """HTML：丢标签留文字，标题与段落边界尽量保住。"""
    from html.parser import HTMLParser

    HEADS = ("h1", "h2", "h3", "h4", "h5", "h6")

    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.out = []
            self.buf = []
            self.skip = 0
            self.cur_head = ""
            self.in_head = False
            self.head_buf = []

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "nav", "footer", "svg"):
                self.skip += 1
            if tag in HEADS:
                self.flush()
                self.in_head = True
                self.head_buf = []
            elif tag in ("p", "div", "li", "tr", "br", "section", "article", "td"):
                self.buf.append("\n")

        def handle_endtag(self, tag):
            if tag in ("script", "style", "nav", "footer", "svg") and self.skip:
                self.skip -= 1
            if tag in HEADS and self.in_head:
                self.in_head = False
                self.cur_head = _clean(" ".join(self.head_buf))

        def handle_data(self, data):
            if self.skip:
                return
            if self.in_head:
                self.head_buf.append(data)
            else:
                self.buf.append(data)

        def flush(self):
            body = _clean("".join(self.buf))
            if body:
                self.out.append((_clean(self.cur_head), body))
            self.buf = []

    p = P()
    p.feed(text)
    p.flush()
    blocks = p.out or blocks_text(re.sub(r"<[^>]+>", " ", text))
    return [(h, t) for h, t in blocks if t.strip()]


def blocks_pdf(path: str):
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise MissingDep("读取 PDF 需要 pypdf：.venv/bin/pip install pypdf") from exc
    reader = PdfReader(path)
    out = []
    for n, page in enumerate(reader.pages, 1):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        t = _clean(t)
        if t:
            out.append((f"第 {n} 页", t))
    return out


def blocks_docx(path: str):
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover
        raise MissingDep("读取 DOCX 需要 python-docx：.venv/bin/pip install python-docx") from exc
    d = docx.Document(path)
    out, buf, head = [], [], ""
    for p in d.paragraphs:
        t = (p.text or "").strip()
        style = (p.style.name or "").lower() if p.style is not None else ""
        if style.startswith("heading") or style.startswith("标题"):
            body = _clean("\n".join(buf))
            if body:
                out.append((head, body))
            buf, head = [], t
            continue
        if t:
            buf.append(t)
    body = _clean("\n".join(buf))
    if body:
        out.append((head, body))
    for tbl in d.tables:
        rows = [" | ".join(c.text.strip() for c in r.cells) for r in tbl.rows]
        txt = _clean("\n".join(rows))
        if txt:
            out.append((head, txt))
    return out


def blocks_notebook(path: str, text: str):
    try:
        nb = json.loads(text)
    except ValueError:
        return blocks_text(text)
    out = []
    for n, cell in enumerate(nb.get("cells") or [], 1):
        src = cell.get("source") or []
        body = _clean("".join(src) if isinstance(src, list) else str(src))
        if body:
            out.append((f"{cell.get('cell_type', 'cell')} #{n}", body))
    return out


def read_blocks(path: str, doc_type: str | None = None):
    """返回 [(标题路径, 正文)]。读不动的抛异常，由上层记一条失败。"""
    doc_type = doc_type or detect_type(path) or "txt"
    if doc_type == "pdf":
        return blocks_pdf(path)
    if doc_type == "docx":
        return blocks_docx(path)
    with open(path, "rb") as fh:
        raw = fh.read()
    if _looks_binary(raw):
        raise ValueError("疑似二进制文件，已跳过")
    text = _decode(raw)
    if doc_type == "md":
        return blocks_markdown(text)
    if doc_type == "html":
        return blocks_html(text)
    if doc_type == "csv":
        return blocks_csv(path, text)
    if doc_type == "notebook":
        return blocks_notebook(path, text)
    if doc_type == "json":
        try:
            return [("", _clean(json.dumps(json.loads(text), ensure_ascii=False, indent=2)))]
        except ValueError:
            return blocks_text(text)
    if doc_type == "code":
        return blocks_code(text)
    return blocks_text(text)


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "big5"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _looks_binary(raw: bytes) -> bool:
    head = raw[:4096]
    if b"\x00" in head:
        return True
    # 高位字节占比过高，多半是图片/压缩包
    if head and sum(1 for b in head if b > 127) / len(head) > 0.9:
        try:
            head.decode("utf-8")
            return False
        except UnicodeDecodeError:
            return True
    return False


# --------------------------------------------------------------------------- #
# 段落 → 片段
# --------------------------------------------------------------------------- #

SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])\s*|(?<=\.)\s+")


def _split_oversize(text: str, size: int, overlap: int):
    """超长段落按句号切，再按目标长度打包，块间带 overlap 字上下文。"""
    sents = [s for s in SENT_SPLIT.split(text) if s and s.strip()]
    if len(sents) <= 1:
        step = max(1, size - overlap)
        return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]
    out, buf, n = [], [], 0
    for s in sents:
        if n + len(s) > size and buf:
            out.append("".join(buf).strip())
            tail = "".join(buf)[-overlap:] if overlap else ""
            buf, n = ([tail] if tail else []), len(tail)
        buf.append(s)
        n += len(s)
    if buf:
        out.append("".join(buf).strip())
    return [c for c in out if c]


def chunk_blocks(blocks, size: int = 700, overlap: int = 120):
    """把段落打包成目标长度的片段，同一标题下的段落优先留在同一块里。"""
    size = max(200, int(size))
    overlap = max(0, min(int(overlap), size // 2))
    chunks = []
    buf, buf_head, buf_len = [], "", 0

    def flush():
        nonlocal buf, buf_len
        body = "\n\n".join(b for b in buf if b).strip()
        if body:
            chunks.append({"heading": buf_head, "text": body})
        buf, buf_len = [], 0

    for head, body in blocks:
        body = body.strip()
        if not body:
            continue
        if buf and (buf_head != head or buf_len + len(body) > size):
            flush()
        buf_head = head
        if len(body) > size * 1.2:
            flush()
            for piece in _split_oversize(body, size, overlap):
                chunks.append({"heading": head, "text": piece})
            continue
        buf.append(body)
        buf_len += len(body)
        if buf_len >= size:
            flush()
    flush()

    # 去掉纯空白与重复（同一文档里完全相同的块只留一个）
    seen, out = set(), []
    for c in chunks:
        key = (c["heading"], c["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def text_blocks_for_text(text: str):
    """外部直接给的一段文字：按 markdown 解析，失败就按段落。"""
    try:
        blocks = blocks_markdown(text)
    except Exception:
        blocks = []
    return blocks or blocks_text(text)


__all__ = [
    "detect_type", "is_candidate", "iter_files", "read_blocks", "chunk_blocks",
    "text_blocks_for_text", "MissingDep", "SKIP_DIRS", "TYPE_BY_EXT",
]
