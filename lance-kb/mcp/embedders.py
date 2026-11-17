#!/usr/bin/env python3
"""向量化后端：自适应选择，全部走本地/自带能力，不强制联网。

优先级（auto）：
    1. Ollama        —— 本机已跑 ollama 时最好：bge-m3 / nomic-embed-text，中文强、零 Python 依赖
    2. fastembed     —— 装了 fastembed 就用 ONNX 跑 BAAI/bge-small-zh-v1.5，CPU 上也很快
    3. OpenAI 兼容   —— 环境里有 API key / base_url 时走远端
    4. hash          —— 兜底：字符 n-gram 哈希向量，永远可用，但没有语义泛化能力

显式指定：环境变量 LANCE_KB_EMBED = auto|ollama|fastembed|openai|hash

关键约束：**维度一旦定下就不能变**，因为 LanceDB 的向量列是定长列表。
换后端要跑 kb_reindex（服务端会检查维度不一致并给出明确提示，不会默默写坏数据）。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request

DEFAULT_OLLAMA_MODEL = os.environ.get("LANCE_KB_OLLAMA_MODEL") or "bge-m3"
DEFAULT_FASTEMBED_MODEL = os.environ.get("LANCE_KB_ST_MODEL") or "BAAI/bge-small-zh-v1.5"
DEFAULT_OPENAI_MODEL = os.environ.get("LANCE_KB_OPENAI_MODEL") or "text-embedding-3-small"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"


class EmbedError(RuntimeError):
    pass


class Embedder:
    name = "base"
    model = ""
    dim = 0

    def encode(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError

    def info(self) -> dict:
        return {"name": self.name, "model": self.model, "dim": self.dim}

    def describe(self) -> str:
        return f"{self.name}/{self.model} · {self.dim} 维"


# --------------------------------------------------------------------------- #
# 1. Ollama
# --------------------------------------------------------------------------- #

class OllamaEmbedder(Embedder):
    name = "ollama"

    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.host = host.rstrip("/")
        self.dim = 0
        probe = self._embed_api(["_probe"])
        self.dim = len(probe[0])

    def _post(self, path: str, payload: dict, timeout: float = 60.0) -> dict:
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _embed_api(self, texts: list[str]) -> list[list[float]]:
        # /api/embed 是 0.1.23+ 的批量接口；老版本只有 /api/embeddings（一次一条）
        try:
            data = self._post("/api/embed", {"model": self.model, "input": texts})
            vecs = data.get("embeddings") or []
            if len(vecs) == len(texts) and vecs:
                return [[float(x) for x in v] for v in vecs]
            raise EmbedError("ollama /api/embed 返回为空")
        except urllib.error.HTTPError as exc:
            if exc.code not in (404, 400):
                raise EmbedError(f"ollama 返回 {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise EmbedError(f"连不上 ollama（{self.host}）：{exc.reason}") from exc

        out = []
        for t in texts:
            data = self._post("/api/embeddings", {"model": self.model, "prompt": t}, timeout=120)
            out.append([float(x) for x in (data.get("embedding") or [])])
        if not out or not out[0]:
            raise EmbedError("ollama /api/embeddings 返回为空")
        return out

    def encode(self, texts):
        out = []
        for i in range(0, len(texts), 16):
            batch = [t or " " for t in texts[i:i + 16]]
            out.extend(_l2(self._embed_api(batch)))
            if i and i % 160 == 0:
                time.sleep(0)  # 让出 GIL，长任务时前端不至于像卡死
        return out


# --------------------------------------------------------------------------- #
# 2. fastembed（ONNX，无需 torch）
# --------------------------------------------------------------------------- #

class FastEmbedder(Embedder):
    name = "fastembed"

    def __init__(self, model: str = DEFAULT_FASTEMBED_MODEL):
        _ensure_hf_endpoint()
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover
            raise EmbedError("fastembed 未安装") from exc
        self.model = model
        cache = _fastembed_cache_dir()
        prepared = _prepared_model_dir(model)
        try:
            if prepared:
                # 已经逐文件下好了，别再去撞一次镜像的 429
                self._model = TextEmbedding(model_name=model, cache_dir=cache,
                                            specific_model_path=prepared)
            else:
                self._model = TextEmbedding(model_name=model, cache_dir=cache)
        except Exception as first:
            try:
                prepared = _prepare_fastembed_model(model)
                self._model = TextEmbedding(model_name=model, cache_dir=cache,
                                            specific_model_path=prepared)
            except Exception as exc:
                raise EmbedError(f"fastembed 加载模型失败：{exc}") from exc
        vec = list(self._model.embed(["probe"]))[0]
        self.dim = len(vec)

    def encode(self, texts):
        out = []
        for chunk in _batched([t or " " for t in texts], 32):
            out.extend(_l2([list(map(float, v)) for v in self._model.embed(chunk)]))
        return out


# --------------------------------------------------------------------------- #
# 3. OpenAI 兼容接口（含 DashScope / 硅基流动 / 本地 vLLM 等）
# --------------------------------------------------------------------------- #

class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(self, model: str = DEFAULT_OPENAI_MODEL,
                 base: str | None = None, key: str | None = None):
        self.model = model
        self.base = (base or os.environ.get("LANCE_KB_OPENAI_BASE")
                     or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.key = key or os.environ.get("LANCE_KB_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        if not self.key:
            raise EmbedError("缺少 OPENAI_API_KEY / LANCE_KB_OPENAI_KEY")
        vec = self._call(["probe"])
        self.dim = len(vec[0])

    def _call(self, texts):
        payload = {"model": self.model, "input": texts, "encoding_format": "float"}
        req = urllib.request.Request(
            f"{self.base}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            raise EmbedError(f"embedding 接口 {exc.code}：{body}") from exc
        except urllib.error.URLError as exc:
            raise EmbedError(f"连不上 embedding 接口（{self.base}）：{exc.reason}") from exc
        items = sorted(data.get("data") or [], key=lambda d: d.get("index", 0))
        vecs = [[float(x) for x in d["embedding"]] for d in items]
        if not vecs:
            raise EmbedError("embedding 接口返回为空")
        return vecs

    def encode(self, texts):
        out = []
        for batch in _batched([t or " " for t in texts], 32):
            out.extend(_l2(self._call(batch)))
        return out


# --------------------------------------------------------------------------- #
# 4. 兜底：字符 n-gram 哈希
# --------------------------------------------------------------------------- #

class HashEmbedder(Embedder):
    """没有模型可用时的兜底。

    做法是把字符 2/3-gram（中文）与词（英文）哈希到固定维度并做符号散列，
    等价于一个固定的随机投影 —— 字面相近的文本向量也相近，同义改写则抓不住。
    所以它必须配合全文检索（hybrid）用，单靠它召回质量有限，这点在 stats 里会说明。
    """

    name = "hash"
    model = "char-ngram-hash"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts):
        out = []
        for t in texts:
            out.append(self._one(t or ""))
        return out

    def _one(self, text: str):
        v = [0.0] * self.dim
        s = re.sub(r"\s+", " ", text.lower().strip())
        if not s:
            return v
        # 中文按字符 n-gram，英文按词
        for n in (2, 3):
            for i in range(len(s) - n + 1):
                gram = s[i:i + n]
                if gram.strip() == "":
                    continue
                self._add(v, gram, 1.0 if n == 2 else 0.7)
        for word in re.findall(r"[a-z0-9_]{2,}", s):
            self._add(v, "w:" + word, 1.5)
        for token in re.findall(r"[a-z0-9_]+", s):
            self._add(v, "t:" + token, 0.5)
        return _unit(v)

    def _add(self, v, key: str, weight: float):
        h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if h[4] & 1 else -1.0
        v[idx] += sign * weight


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #

def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _l2(vecs: list[list[float]]) -> list[list[float]]:
    return [_unit(v) for v in vecs]


def ollama_available(host: str = OLLAMA_HOST, timeout: float = 1.2) -> bool:
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _fastembed_cache_dir() -> str:
    return os.environ.get("LANCE_KB_MODEL_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "lance-kb")


def _fastembed_model_cached(model: str = DEFAULT_FASTEMBED_MODEL) -> bool:
    """模型是否已经落到本地 —— stats 里要区分「能离线跑」和「下次要下 100MB」。"""
    needle = model.split("/")[-1].lower()
    try:
        if any(needle in name.lower() for name in os.listdir(_fastembed_cache_dir())):
            return True
    except Exception:
        pass
    return bool(_prepared_model_dir(model))


def _model_spec(model: str) -> dict:
    """从 fastembed 自己的模型清单里取 hf 仓库名和 onnx 文件名。"""
    try:
        from fastembed import TextEmbedding
        return next((m for m in TextEmbedding.list_supported_models()
                     if m.get("model") == model), {}) or {}
    except Exception:
        return {}


def _prepared_model_dir(model: str) -> str:
    """逐文件下载好的模型目录（存在且 onnx 就位才算数）。"""
    spec = _model_spec(model)
    name = spec.get("model_file")
    if not name:
        return ""
    dest = os.path.join(_fastembed_cache_dir(), "models", model.replace("/", "--"))
    path = os.path.join(dest, name)
    try:
        if os.path.getsize(path) > 1024:
            return dest
    except Exception:
        pass
    return ""


def _prepare_fastembed_model(model: str) -> str:
    """绕开 fastembed 自带的下载器，逐个文件下到固定目录，返回该目录。

    为什么需要这一层：fastembed 默认走 hf 的 snapshot_download，会先调
    `/api/models/<repo>` 查仓库元数据 —— 国内镜像对这个端点限流（429），而且
    部分 ONNX 仓库里只有 `model_optimized.onnx`、没有它 model_file 里写的文件名，
    整体就会抛 "Could not load model from any source"。逐文件下 + specific_model_path
    这两点都能绕过去。
    """
    from huggingface_hub import hf_hub_download

    spec = _model_spec(model)
    repo = (spec.get("sources") or {}).get("hf")
    model_file = spec.get("model_file")
    if not repo or not model_file:
        raise EmbedError(f"fastembed 清单里找不到 {model} 的下载源，无法预取")

    dest = os.path.join(_fastembed_cache_dir(), "models", model.replace("/", "--"))
    os.makedirs(dest, exist_ok=True)
    names = ["config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "vocab.txt", model_file]
    names += list(spec.get("additional_files") or [])
    for name in names:
        target = os.path.join(dest, name)
        if os.path.exists(target) and os.path.getsize(target) > 0:
            continue
        try:
            hf_hub_download(repo_id=repo, filename=name, local_dir=dest)
        except Exception:
            # 可选文件（vocab.txt 之类）缺了不影响加载，关键看 onnx 下没下来
            continue
    if not os.path.exists(os.path.join(dest, model_file)):
        raise EmbedError(
            f"模型 {model} 下载失败（源 {repo}）。国内可先 export HF_ENDPOINT=https://hf-mirror.com 再试。")
    return dest


def _ensure_hf_endpoint() -> None:
    """国内机器直连 huggingface.co 基本不通，首次下 fastembed 模型会卡死。

    只在用户没显式设 HF_ENDPOINT 时探测：能直连就不动，直连不通而镜像可达就切镜像。
    两步都有超时，探测失败也不拦着 —— 让 fastembed 自己把错误抛出来更清楚。
    """
    if os.environ.get("HF_ENDPOINT"):
        return
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=1.5).close()
        return
    except Exception:
        pass
    try:
        urllib.request.urlopen("https://hf-mirror.com", timeout=1.5).close()
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    except Exception:
        pass


def _try(factory):
    try:
        return factory()
    except EmbedError:
        return None
    except Exception:
        return None


def resolve(prefer: str | None = None) -> Embedder:
    """按优先级挑一个能用的后端；全部失败时回落到 hash（永远不会失败）。"""
    prefer = (prefer or os.environ.get("LANCE_KB_EMBED") or "auto").lower()

    if prefer == "hash":
        return HashEmbedder()
    if prefer == "ollama":
        emb = _try(lambda: OllamaEmbedder())
        if emb:
            return emb
        raise EmbedError(f"指定了 ollama，但 {OLLAMA_HOST} 不可用")
    if prefer == "fastembed":
        emb = _try(lambda: FastEmbedder())
        if emb:
            return emb
        raise EmbedError("指定了 fastembed，但加载失败（.venv/bin/pip install fastembed）")
    if prefer == "openai":
        emb = _try(lambda: OpenAIEmbedder())
        if emb:
            return emb
        raise EmbedError("指定了 openai，但缺少 key 或接口不可用")

    # auto：按顺序试，谁先能用就用谁
    if (os.environ.get("LANCE_KB_EMBED_ORDER") or "").strip():
        order = [x.strip() for x in os.environ["LANCE_KB_EMBED_ORDER"].split(",") if x.strip()]
    else:
        order = ["ollama", "fastembed", "openai", "hash"]
    for name in order:
        if name == "ollama":
            if not ollama_available():
                continue
            emb = _try(lambda: OllamaEmbedder())
            if emb:
                return emb
        elif name == "fastembed":
            emb = _try(lambda: FastEmbedder())
            if emb:
                return emb
        elif name == "openai":
            if not (os.environ.get("LANCE_KB_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")):
                continue
            emb = _try(lambda: OpenAIEmbedder())
            if emb:
                return emb
        elif name == "hash":
            return HashEmbedder()
    return HashEmbedder()


def available_backends() -> dict:
    """给 kb_stats / 自检用：报告每个后端能不能用。"""
    out = {}
    out["ollama"] = {"available": ollama_available(),
                     "detail": f"{OLLAMA_HOST} · model={DEFAULT_OLLAMA_MODEL}"}
    try:
        import fastembed  # noqa: F401
        if _fastembed_model_cached():
            detail = f"model={DEFAULT_FASTEMBED_MODEL}（已下载，可离线）"
        else:
            detail = f"model={DEFAULT_FASTEMBED_MODEL}（首次使用要下约 100MB）"
        out["fastembed"] = {"available": True, "detail": detail}
    except ImportError:
        out["fastembed"] = {"available": False, "detail": "未安装：pip install fastembed"}
    key = os.environ.get("LANCE_KB_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    out["openai"] = {"available": bool(key),
                     "detail": f"model={DEFAULT_OPENAI_MODEL} base={os.environ.get('LANCE_KB_OPENAI_BASE') or 'api.openai.com'}"}
    out["hash"] = {"available": True, "detail": "兜底：512 维字符 n-gram 哈希，必须配合全文检索"}
    try:
        import jieba  # noqa: F401
        out["jieba"] = {"available": True, "detail": "中文分词已启用，全文索引走词级"}
    except ImportError:
        out["jieba"] = {"available": False, "detail": "未安装 jieba，全文索引退回 ngram(2-3)"}
    return out


__all__ = ["resolve", "available_backends", "Embedder", "EmbedError",
           "HashEmbedder", "OllamaEmbedder", "FastEmbedder", "OpenAIEmbedder"]
