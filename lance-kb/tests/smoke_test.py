#!/usr/bin/env python3
"""端到端冒烟测试：在临时目录里建库、入库、检索、同步、删除，不碰真实知识库。

    .venv/bin/python tests/smoke_test.py

断言的是「行为」而不是「数字」：换了嵌入后端、改了切块参数，这些断言仍然成立。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "mcp"))

TMP = tempfile.mkdtemp(prefix="lance-kb-test-")
os.environ["LANCE_KB_HOME"] = TMP
os.environ.setdefault("LANCE_KB_EMBED", "hash")  # 测试不依赖模型，跑得快

import kb  # noqa: E402

FAILED = []


def check(name: str, cond: bool, detail: str = ""):
    if cond:
        print(f"  ✓ {name}")
    else:
        print(f"  × {name}  {detail}")
        FAILED.append(name)


def write(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def main() -> int:
    corpus = os.path.join(TMP, "corpus")
    write(os.path.join(corpus, "deploy.md"), """# 部署手册

## 灰度发布

灰度发布分三步：先把 5% 流量切到新版本，观察 30 分钟错误率，再逐步放大到 50%、100%。
回滚方式是把流量权重切回旧版本，不需要重启服务。

## 数据库迁移

迁移脚本放在 migrations/ 目录，命名必须带时间戳前缀，例如 20260101_add_index.sql。
""")
    write(os.path.join(corpus, "notes", "meeting-2026-03.md"), """# 三月周会

## 结论

知识库的切片长度定在 700 字，重叠 120 字。检索先用混合检索，召回不够再调关键词。
""")
    write(os.path.join(corpus, "config.yaml"), "retention_days: 30\nindex_batch: 64\n")

    print("\n[1] 入库")
    r = kb.add_paths([corpus], tags="测试,部署")
    check("三个文件都入库", len(r["added"]) == 3, str([a["name"] for a in r["added"]]))
    check("生成了片段", r["chunks"] > 0, f"chunks={r['chunks']}")
    check("文档总数正确", kb.stats()["docs"] == 3)

    print("\n[2] 增量：内容没变应全部跳过")
    r2 = kb.add_paths([corpus])
    check("重复入库全部跳过", len(r2["added"]) == 0 and len(r2["skipped"]) == 3,
          f"added={len(r2['added'])} skipped={len(r2['skipped'])}")

    print("\n[3] 混合检索")
    res = kb.search("灰度发布怎么回滚", k=5)
    check("有命中", bool(res["hits"]))
    top = res["hits"][0] if res["hits"] else {}
    check("命中部署手册", "deploy" in (top.get("doc_name") or ""), str(top.get("doc_name")))
    check("带小节标题", "灰度" in (top.get("heading") or ""), str(top.get("heading")))
    check("两路都有名次", set(top.get("ranks", {})) >= {"vector", "keyword"},
          str(top.get("ranks")))

    print("\n[4] 关键词检索（专有名词）")
    kw = kb.search("migrations", k=3, mode="keyword")
    check("keyword 命中", bool(kw["hits"]))
    check("命中的是部署手册",
          any("deploy" in (h.get("doc_name") or "") for h in kw["hits"]),
          str([h.get("doc_name") for h in kw["hits"]]))

    print("\n[5] 过滤")
    only_notes = kb.search("切片长度", k=5, filters={"path_prefix": os.path.join(corpus, "notes")})
    check("路径前缀过滤生效",
          all("notes" in (h.get("doc_path") or "") for h in only_notes["hits"]))

    print("\n[6] 取上下文")
    cid = top.get("chunk_id")
    ctx = kb.get_chunk(cid, before=1, after=1)
    check("拿到文档元信息", bool(ctx.get("doc")))
    check("包含焦点片段", any(c["is_focus"] for c in ctx.get("chunks") or []))

    print("\n[7] 清单与标签")
    docs = kb.list_docs(limit=10)
    check("清单返回 3 个", docs["total"] == 3, str(docs["total"]))
    check("按目录加了标签", kb.list_docs(limit=10, filters={"tags": "部署"})["total"] == 3,
          str(kb.list_docs(limit=10, filters={"tags": "部署"})["total"]))
    kb.add_paths([os.path.join(corpus, "notes")], tags="笔记")
    check("覆盖标签后过滤生效",
          kb.list_docs(limit=10, filters={"tags": "笔记"})["total"] == 1,
          str(kb.list_docs(limit=10, filters={"tags": "笔记"})["total"]))

    print("\n[8] 同步：改一个、加一个、删一个")
    write(os.path.join(corpus, "deploy.md"), "# 部署手册\n\n## 灰度发布\n\n改成两步：5% 然后 100%。\n")
    write(os.path.join(corpus, "notes", "new.md"), "# 新笔记\n\n这是同步时才出现的新文件。\n")
    os.remove(os.path.join(corpus, "config.yaml"))
    s = kb.sync()
    check("同步新增 1", s["added"] == 1, str(s["added"]))
    check("同步更新 1", s["updated"] == 1, str(s["updated"]))
    check("清理已删除 1", len(s["pruned"]) == 1, str(s["pruned"]))
    check("清理后文档数 3", s["total_docs"] == 3, str(s["total_docs"]))

    print("\n[9] 重建索引")
    ri = kb.reindex()
    check("重建覆盖全部文档", ri["docs"] == 3, str(ri["docs"]))
    check("重建后仍可检索", bool(kb.search("灰度", k=3)["hits"]))

    print("\n[10] 看板载荷")
    payload = kb.dashboard_payload()
    check("载荷有文档", len(payload["docs"]) == 3)
    check("载荷有片段", len(payload["chunks"]) > 0)
    check("meta 完整", all(k in payload["meta"] for k in ("kb_home", "docs_total", "embedder")))

    print("\n[11] 删除")
    rm = kb.remove(path_prefix=corpus)
    check("全部删除", rm["removed"] == 3, str(rm["removed"]))
    check("删完为空", kb.stats()["docs"] == 0)

    print("\n[12] 空库行为")
    empty = kb.search("随便问问")
    check("空库给出提示而不是报错", empty.get("empty") is True, str(empty))
    add_empty = kb.add_paths([os.path.join(TMP, "不存在")])
    check("空路径给出失败说明", bool(add_empty["failed"]), str(add_empty))

    print()
    if FAILED:
        print(f"× {len(FAILED)} 项失败：" + "、".join(FAILED))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
