# local-lance-kb

一个**完全跑在本机**的 WorkBuddy 知识库插件市场。核心插件 `lance-kb` 把你的文档切块、向量化、存进
[LanceDB](https://lancedb.com/)，检索时**向量召回 + 全文召回两路融合**，回答带出处。资料不出机器，断网也能用。

> 这是一个插件市场仓库（marketplace），不是一个可执行程序。插件本体在 [`lance-kb/`](./lance-kb)，
> 完整用法、参数、坑与实现细节见 **[lance-kb/README.md](./lance-kb/README.md)**。

## 能做什么

- **收**：`md / txt / pdf / docx / html / csv / json / ipynb` 与常见代码文件，目录递归，按内容哈希增量
- **存**：LanceDB 单目录，`docs` 记文档、`chunks` 记片段与向量，跟着源码目录走，随时能删
- **查**：9 个 MCP 工具 —— 混合检索、上下文回取、清单、删除、重建、增量同步
- **看**：库看板直接内嵌在对话里（类型分布、标签、文档清单、片段浏览），**还能在看板里拖文件入库**
- **中文**：默认用中文向量模型 + jieba 词级索引，不是拿英文分词器硬切中文
- **引用**：每个片段都带「文档名 › 小节」，结论可追溯

## 快速开始

```bash
git clone git@github.com:leishu/local-lance-kb.git
cd local-lance-kb/lance-kb

./install.sh --dry-run    # 先看它会改哪些文件
./install.sh              # 建 .venv、装依赖、注册到本地市场
```

装完 **完全退出 WorkBuddy 再打开**（`enabledPlugins` 只在启动时读），首次调用会弹「信任此 MCP 服务」，点信任。

然后在对话里说：

> 把 `~/work/notes` 收进本地知识库，标签「工作笔记」
> 我们部署规范里灰度发布是怎么规定的？在我的资料里找找
> 打开知识库看板

## 仓库结构

```
local-lance-kb/
├── .codebuddy-plugin/marketplace.json   市场清单（市场名 lance-kb-local）
├── README.md                            本文件：仓库入口
└── lance-kb/                            插件本体
    ├── .codebuddy-plugin/plugin.json    插件清单（MCP 服务注册）
    ├── mcp/
    │   ├── server.py      MCP 服务端（9 个工具 + 看板资源）
    │   ├── kb.py          数据层：入库、切块、检索、统计、删除、同步
    │   ├── embedders.py   嵌入后端自适应（ollama / fastembed / openai / hash）
    │   ├── loaders.py     各格式解析 + 通用切块
    │   └── launch.sh      解释器探测与启动
    ├── dashboard/index.html              单文件看板（无构建步骤）
    ├── skills/lance-kb/SKILL.md          检索策略与回答口径
    ├── commands/收录资料.md
    ├── tests/smoke_test.py               端到端自检（临时目录建库）
    ├── install.sh                        装依赖 + 注册
    └── README.md                         详细文档
```

## 技术要点

- **中文全文检索**：LanceDB 0.4x 的 native FTS 没有 jieba，`simple` 会把整段中文当一个 token，`icu` 又不支持
  `language="Chinese"`。做法是入库时用 jieba 切词写进 `search_tokens` 列，索引建在该列并指定
  `base_tokenizer="whitespace"` —— 对 tantivy 来说每个词就是一个 token，BM25 直接在中文词上算。
- **融合**：向量与全文各取 top 4k，用 RRF（`score = Σ 1/(60+rank)`）合并，只看名次不看原始分数，
  天然避开「余弦距离」和「BM25 分数」量纲不同的问题。筛选条件下推到两路召回，不是取回后再过滤。
- **中文向量**：默认 fastembed + `BAAI/bge-small-zh-v1.5`（512 维 ONNX，纯 CPU）。装了 Ollama 则优先用 `bge-m3`。
- **后端身份校验**：`hash` 与 `bge-small-zh` 都是 512 维，所以比对的是 config 里的 `name/model` 而不是维度，
  避免换后端后新旧向量混着查。

## 环境注意

- `install.sh` 会建 `lance-kb/.venv`，装了 onnxruntime 后约 **600MB+**，PyPI 慢时首次要等半小时。
- 如果 `/tmp` 很小（比如 10MB tmpfs），`pip` 解包大 wheel 会报「设备上没有空间」。
  先 `export TMPDIR=/path/to/big/dir`，`install.sh` 也内置了 `.pip-tmp` 兜底。
- `huggingface.co` 不通时，插件会自动探测并切到 `hf-mirror.com`；镜像的 `/api/models` 端点会 429，
  所以模型是逐文件 `hf_hub_download` 下来的，不走 `snapshot_download`。

## 运行时不入库的东西

`.venv/`、`lance-kb/data/`（LanceDB 表与 config）、`.workbuddy/`、`.tmp/`、`.idea/` 都在 `.gitignore` 里 ——
每个人的知识库与依赖各自生成，不进版本库。

## 许可

MIT。
