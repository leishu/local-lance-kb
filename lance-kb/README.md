# lance-kb · 本地知识库

把自己的文档收进本机，用 [LanceDB](https://lancedb.com/) 存片段与向量，检索时**向量召回 + 全文召回
两路融合**，答案带出处。资料不出机器，断网也能用。

- **收**：`md / txt / pdf / docx / html / csv / json / ipynb / 常见代码文件`，目录递归，按内容哈希增量
- **存**：LanceDB 单目录（`data/lancedb/`），`docs` 表记文档、`chunks` 表记片段与向量
- **查**：9 个 MCP 工具 —— 混合检索、上下文回取、清单、删除、重建、增量同步
- **看**：库看板内嵌在对话里（类型分布、标签、文档清单、片段浏览），单文件 HTML 无需构建
- **引用**：每个片段都带「文档名 › 小节」，回答可追溯

## 一、安装

插件通过本地市场注册，不打包、不分发，改代码重启即生效：

```bash
mkdir -p ~/lance-market && cd ~/lance-market
# 把本目录放到 ~/lance-market/lance-kb（目录名必须是 lance-kb）
cd lance-kb
./install.sh --dry-run      # 先看会改什么
./install.sh                # 建 .venv、装依赖、注册插件
```

`install.sh` 做两件事：

1. 建 `lance-kb/.venv` 并装 `lancedb pypdf python-docx`（约 100~150MB，首次较慢；
   想换国内镜像就先 `export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`）
2. 改 4 个注册表位置（每个都先备份 `.bak-<时间戳>`）：市场清单、`known_marketplaces.json`、
   `installed_plugins.json`、`settings.json` 的 `enabledPlugins`，并清掉 `mcp.json` 里的同名旧服务

然后**完全退出 WorkBuddy 再打开**（`enabledPlugins` 只在启动时读），首次调用会弹「信任此 MCP 服务」，
点信任。

只想装依赖、不碰注册表：`./install.sh --deps-only`。

装完想确认「入库—检索—同步—删除」整条链路是通的，跑一遍自检（在临时目录里建库，不动真实数据）：

```bash
.venv/bin/python tests/smoke_test.py
```

## 二、中文向量与中文分词（决定检索质量，先看这段）

检索质量一半靠切块、一半靠向量和分词。默认自动挑本机最能用的组合，不用手动配。

### 向量：优先中文模型

| 后端 | 怎么来的 | 适合 |
|---|---|---|
| `ollama` | 本机已跑 Ollama，默认模型 `bge-m3` | **推荐**。中文最强，零 Python 依赖：`ollama pull bge-m3` |
| `fastembed` | `./install.sh --with-fastembed`，ONNX 跑 `BAAI/bge-small-zh-v1.5`（512 维） | 不想装 Ollama 就用它。CPU 上一条片段几十毫秒，模型约 90MB 只需下一次；直连 HF 不通会自动切 hf-mirror |
| `openai` | 环境里有 `OPENAI_API_KEY` / `LANCE_KB_OPENAI_KEY` | 有云端 key，且不在意出网 |
| `hash` | 内置兜底，字符 n-gram 哈希 512 维 | **永远可用，但没有语义泛化能力**，只能靠全文检索那一路 |

### 分词：jieba 词级索引

`install.sh` 默认装 `jieba`。装上之后全文索引走**真正的中文分词**：入库时把 jieba
切好的词用空格连成一列 `search_tokens`，索引建在它上面并指定 `base_tokenizer="whitespace"`
—— 对 tantivy 来说每个词就是一个 token，BM25 就是在中文词上算的。查询走同一个切词函数，
两端规则必须一致，否则分数对不上。

为什么不直接用 LanceDB 的分词器：0.4x 的 native FTS 只认 simple/raw/whitespace/ngram/icu，
**没有 jieba**；`simple` 按空白切词，整段中文会变成一个 token（等于搜不到）；`icu` 又不支持
`language="Chinese"`。所以没装 jieba 时只能退回 `ngram(2-3)` 字符切片——能用，但索引更大、
精度更差（`发票原件` 这种会命中很多无关的二字片段）。`kb_stats` 会显示当前用的是 `jieba`
还是 `ngram(2-3)`，两者切换需要 `kb_reindex`。

强制指定：`LANCE_KB_EMBED=ollama`（或 `fastembed` / `openai` / `hash`）。相关环境变量：

```bash
LANCE_KB_EMBED=auto              # auto|ollama|fastembed|openai|hash
LANCE_KB_EMBED_ORDER=ollama,hash # auto 的尝试顺序
LANCE_KB_FTS=auto                # auto|jieba|ngram —— 全文分词方案
LANCE_KB_OLLAMA_MODEL=bge-m3
OLLAMA_HOST=http://127.0.0.1:11434
LANCE_KB_ST_MODEL=BAAI/bge-small-zh-v1.5
LANCE_KB_MODEL_CACHE=~/.cache/lance-kb  # fastembed 的模型下载目录
LANCE_KB_OPENAI_BASE=https://api.deepseek.com/v1   # 兼容任何 OpenAI 协议端点
LANCE_KB_OPENAI_KEY=sk-...
LANCE_KB_HOME=/path/to/data      # 知识库放在哪，默认插件下 data/
HF_ENDPOINT=https://hf-mirror.com  # 国内下载 fastembed 模型时用
```

**换后端必须重建索引**：向量维度不同时工具会直接报「库里是 X 维，当前是 Y 维」；
维度恰好相同时（`hash` 和 `bge-small-zh` 都是 512 维）则比对后端身份并报「向量空间不同，混着查会串味」。
两种情况都跑 `kb_reindex` 全量重建，不要试图用重新 `kb_add` 蒙过去。库是空的时候会自动按新后端重建，不用管。

## 三、开始用

在对话里说：

> 把 `~/work/notes` 收进本地知识库，标签「工作笔记」
> 我们部署规范里灰度发布是怎么规定的？在我的资料里找找
> 知识库里都有什么？打开看板看看

对应工具：`kb_add` / `kb_search` / `kb_stats` / `open_dashboard`。

命令行（不经过对话，适合脚本与定时任务）：

```bash
.venv/bin/python mcp/kb.py add ~/work/notes --tags 工作笔记
.venv/bin/python mcp/kb.py note --name "周会结论" --text "灰度先放 5% 流量…"
.venv/bin/python mcp/kb.py search -q "灰度发布" -k 8
.venv/bin/python mcp/kb.py search -q "deploy.sh" --mode keyword
.venv/bin/python mcp/kb.py docs --limit 30
.venv/bin/python mcp/kb.py stats
.venv/bin/python mcp/kb.py sync
.venv/bin/python mcp/kb.py reindex
.venv/bin/python mcp/kb.py rm --path-prefix ~/work/notes
```

## 四、MCP 工具

| 工具 | 用途 |
|---|---|
| `kb_stats` | 文档/片段/字数、类型分布、标签、嵌入后端与维度、全文分词器、来源目录 |
| `kb_add` | 入库：`paths`（文件/目录/通配）或 `name`+`text`；`tags`、`force`、`chunk_size` |
| `kb_search` | `mode=hybrid\|vector\|keyword`，可叠 `doc_type` / `tags` / `path_prefix` / `path_contains` |
| `kb_open` | 按 `chunk_id` 取前后片段，或按 `doc_id` 看开头 |
| `kb_docs` | 文档清单，按名字/类型/标签过滤，五种排序 |
| `kb_remove` | 按 `doc_ids` / `path_prefix` / `pattern` / `tags` / `all` 删除 |
| `kb_reindex` | 重建向量与全文索引（换后端、改切块参数后必用） |
| `kb_sync` | 重扫登记过的来源目录，增量更新 + 清理已删除文件 |
| `open_dashboard` | 打开库看板 |

配套 Skill（`skills/lance-kb/SKILL.md`）写了检索策略、引用口径与回答套路，模型处理知识库问题时会自动加载。

## 五、它是怎么切的、怎么查的

**切块**：先按格式解析成 `(小节标题, 正文)` 段落 —— Markdown 按 `#` 层级、PDF 按页、DOCX 按标题样式、
CSV 每 40 行一批且每批重复表头、代码在顶层 `def/class` 处断开。再把段落按目标长度（默认 700 字）
打包，超长段落按句号切开并带 120 字重叠。嵌入时把「小节标题」拼在正文前，标题本身也参与语义。
全文索引建在 `search_tokens`（jieba 切好的词，空格连接）上 —— native FTS 一次只能索引一个字段，
所以标题和正文必须先并到一起再切词；没装 jieba 时退回 `search_text` + `ngram(2-3)`。

**检索**：

```
query ──┬── 向量召回（cosine，取 top 4k）
        └── 全文召回（jieba 切词后查 search_tokens，取 top 4k）
                 ↓
        RRF 融合（score = Σ 1/(60+rank)）→ top k 片段
                 ↓
        片段带 doc_name / heading / ordinal → 可用 kb_open 取上下文
```

RRF 只看名次不看原始分数，天然避开「向量距离」和「BM25 分数」量纲不同的问题。
筛选条件（类型/标签/路径）会下推到两路召回，不是先取回再过滤。

## 六、看板

对 WorkBuddy 说「打开知识库看板」，或直接调 `open_dashboard`。看板是 `dashboard/index.html`
单文件（**没有构建步骤**），服务端把数据以 gzip+base64 注入到 `__LANCE_KB_DATA_B64__` 占位符：

- **KPI**：文档数、片段数、总字数、平均片段长度、切块参数
- **类型分布**：点条目筛选文档清单
- **标签**：入库时传过 `tags` 的才会出现，点选叠加过滤
- **文档清单**：文件名/路径搜索、按更新时间/片段数/字数/名称排序，点一行看它的片段
- **片段浏览**：点片段展开全文
- 顶部徽章显示嵌入后端、全文分词器、磁盘占用；用兜底哈希向量时会标黄提示

### 在看板里直接加文件

看板左上角有「添加文件」：把文件拖进去（或点「选择文件」），填个标签，立刻入库并刷新。
支持多选，单个 ≤ 8MB，格式与 `kb_add` 一致。

怎么做到的：看板是 webview 里的静态 HTML，碰不到磁盘，所以 MCP 服务启动时会额外开一个
**只监听 `127.0.0.1`、端口由系统分配**的小 HTTP 服务，看板把文件内容 POST 过去，服务端落到
`data/inbox/` 再走正常入库流程。每次启动生成一次性随机 token 并注入看板载荷，
别的网页拿不到 token 就写不进来。

- `POST /ingest`：`{token, tags, files:[{name, b64}]}`，返回新增/更新/跳过/片段数
- `GET /payload`：返回最新看板数据（入库后页面用它原地刷新，不用重开）
- `GET /ping`：探活

两点限制：宿主如果给 webview 加了严格的 CSP 拦掉 `127.0.0.1`，看板会明确提示失败，
此时改用对话或命令行入库；上传的文件留在 `data/inbox/`（方便追溯），删文档不影响这些原件。

本地预览（不经过 MCP）：

```bash
.venv/bin/python mcp/kb.py export-dev   # 写 dashboard/data.dev.json
# 然后浏览器打开 dashboard/index.html（未注入时会提示）
```

## 七、目录

```
lance-kb/
├── .codebuddy-plugin/plugin.json   插件清单（MCP 服务注册）
├── mcp/
│   ├── server.py     MCP 服务端（9 个工具 + 看板资源，协议自己实现，不依赖 mcp SDK）
│   ├── kb.py         数据层：入库、切块、检索、统计、删除、同步、命令行
│   ├── embedders.py  嵌入后端自适应（ollama / fastembed / openai / hash）
│   ├── loaders.py    各格式解析 + 通用切块
│   └── launch.sh     解释器探测与启动（找得到 .venv 就用 .venv）
├── dashboard/index.html            单文件看板（无需构建）
├── skills/lance-kb/SKILL.md        检索策略与回答口径
├── commands/收录资料.md
├── tests/smoke_test.py             端到端自检（临时目录里建库，不碰真实知识库）
├── data/                           LanceDB、config.json 与 inbox/（不入库）
├── install.sh                      装依赖 + 注册到本地市场
└── README.md
```

## 八、口径与坑

1. **`hash` 后端不等于语义搜索**。它只是字面相近的随机投影，同义改写抓不住；这种状态下检索靠全文那一路，
   `kb_stats` 会给出 caveat。想要真语义：起 Ollama 或装 fastembed，然后 `kb_reindex`。
2. **jieba 索引和 hash 向量是两件独立的事**。jieba 只决定全文那一路的精度，向量那一路还取决于后端。
   两个都想要最好的：装 fastembed（中文向量 bge-small-zh-v1.5）+ jieba（中文分词），然后 `kb_reindex` 一次。
   注意 `kb_reindex` 会重算向量，换嵌入后端后必须跑。
3. **中文分词有自己的停用词表**（`mcp/kb.py` 里的 `_STOPWORDS`）。查询里"的/怎么/需要"这类词会被挡掉，
   只查纯停用词时会退回未过滤结果。想加行业专有词到 jieba 词典，可以用 `jieba.add_word()`，
   在 `_jieba_mod()` 里加一行即可。
4. **删文件不会自动出库**。要么 `kb_sync`（会清理登记目录下已消失的文件），要么 `kb_remove`。
5. **切块参数改了不等于旧文档变了**。`chunk_size/overlap` 只影响之后入库的文档，要统一就 `kb_reindex`。
6. **PDF 的抽取质量取决于文件本身**。扫描件（纯图片）抽不出文字，会被记为「没有可索引的文本」；
   需要 OCR 的话本插件不做。
7. **同一路径 = 同一文档**。`doc_id` 由绝对路径哈希得到，移动文件等于新增一个文档，旧的要手工删。
8. **库是本地快照**，不跟着源文件实时变。想保鲜就用 `kb_sync` 或建个定时任务：
   > 每天早上 9 点同步本地知识库，并报告新增/更新/清理了多少文档。
9. **隐私**：默认全本地。只有显式配了 `LANCE_KB_EMBED=openai`（或设了 key 让 auto 选中它）时，
   **片段正文会发到那个接口**。用云端 key 前请确认这一点。fastembed 的模型也是本地 ONNX 推理，
   只有首次下载模型会联网。

## 九、许可

MIT。
