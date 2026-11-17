#!/bin/sh
# lance-kb —— 装依赖 + 注册到 WorkBuddy 的「本地插件市场」
#
# 前提：本仓库放在「市场目录」下并叫 lance-kb
#   mkdir -p ~/lance-market && cd ~/lance-market
#   git clone <repo-url> lance-kb
#
# 用法：
#   ./install.sh                  # 建 .venv + 装依赖（含 jieba）+ 注册插件
#   ./install.sh --dry-run        # 只看会改什么，不落盘
#   ./install.sh --deps-only      # 只装 Python 依赖，不动注册表
#   ./install.sh --with-fastembed # 额外装中文向量模型（ONNX 跑 bge-small-zh-v1.5，
#                                 # 首次用会从 HF 下 ~100MB；国内建议先 export
#                                 # HF_ENDPOINT=https://hf-mirror.com）
#   ./install.sh --no-deps        # 跳过依赖安装，只做注册
#
# 会改动 4 个注册表位置（都先备份 .bak-<时间戳>）：
#   <市场目录>/.codebuddy-plugin/marketplace.json      市场清单
#   ~/.workbuddy/plugins/known_marketplaces.json       登记市场
#   ~/.workbuddy/plugins/installed_plugins.json        登记安装
#   ~/.workbuddy/settings.json                         启用（enabledPlugins）
# 另外会清掉 ~/.workbuddy/mcp.json 里的同名旧服务（避免两份配置打架）
#
# 幂等：重复执行只会覆盖同样几个键。

set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
PLUGIN_NAME=$(basename "$SCRIPT_DIR")
MARKET_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
MARKET_NAME=lance-kb-local
VENV="$SCRIPT_DIR/.venv"

say() { printf '%s\n' "$*"; }

if [ ! -f "$SCRIPT_DIR/.codebuddy-plugin/plugin.json" ]; then
    say "× 这里不是插件目录：找不到 .codebuddy-plugin/plugin.json"
    exit 1
fi

PY=
for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PY=$c; break; fi
done
[ -n "$PY" ] || { say "× 需要 python3（用来建虚拟环境与改 JSON 注册表）"; exit 1; }

DRY=0
WITH_FE=0
DEPS=1
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY=1 ;;
        --deps-only) DEPS_ONLY=1 ;;
        --with-fastembed) WITH_FE=1 ;;
        --no-deps) DEPS=0 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) say "未知参数：$arg（可用：--dry-run --deps-only --with-fastembed --no-deps）"; exit 1 ;;
    esac
done

say "插件目录 : $SCRIPT_DIR"
say "市场根   : $MARKET_ROOT"
say "市场名   : $MARKET_NAME"
say "插件名   : $PLUGIN_NAME"
say ""

# --------------------------------------------------------------------------- #
# 1. Python 依赖
# --------------------------------------------------------------------------- #

install_deps() {
    if [ ! -x "$VENV/bin/python" ]; then
        say "→ 创建虚拟环境 $VENV"
        [ "$DRY" = 1 ] && { say "[dry-run] $PY -m venv .venv"; return 0; }
        "$PY" -m venv "$VENV"
    else
        say "→ 复用已有虚拟环境 $VENV"
    fi
    [ "$DRY" = 1 ] && { say "[dry-run] pip install lancedb pypdf python-docx jieba"; return 0; }

    # /tmp 常常是 10MB 的 tmpfs，pip 解包会直接失败，所以把临时目录放到插件下
    mkdir -p "$SCRIPT_DIR/.pip-tmp"
    TMPDIR="$SCRIPT_DIR/.pip-tmp" "$VENV/bin/python" -m pip install -q --upgrade pip || true
    # jieba 是中文分词的关键：没它全文索引只能退回 ngram(2-3) 字符切片
    PKGS="lancedb pypdf python-docx jieba"
    [ "$WITH_FE" = 1 ] && PKGS="$PKGS fastembed"
    say "→ 安装：$PKGS（lancedb 体积较大，首次约 100~150MB，请耐心等）"
    # shellcheck disable=SC2086
    TMPDIR="$SCRIPT_DIR/.pip-tmp" "$VENV/bin/python" -m pip install $PKGS
    rmdir "$SCRIPT_DIR/.pip-tmp" 2>/dev/null || true

    if [ "$WITH_FE" = 1 ]; then
        say "→ 预取中文嵌入模型 bge-small-zh-v1.5（约 90MB；直连不通会自动切 hf-mirror）"
        mkdir -p "$SCRIPT_DIR/.pip-tmp"
        TMPDIR="$SCRIPT_DIR/.pip-tmp" PYTHONPATH="$SCRIPT_DIR/mcp" \
            "$VENV/bin/python" - <<'PY' || true
import os
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
try:
    import embedders
    print("  ✓ 模型就绪：", embedders._prepare_fastembed_model("BAAI/bge-small-zh-v1.5"))
except Exception as exc:
    print("  · 预取没成功（第一次检索时会重试）：", exc)
PY
        rmdir "$SCRIPT_DIR/.pip-tmp" 2>/dev/null || true
    fi

    say "→ 自检："    "$VENV/bin/python" - <<'PY' || true
import sys
print("  Python", sys.version.split()[0])
try:
    import lancedb, pyarrow
    print("  lancedb", lancedb.__version__, "· pyarrow", pyarrow.__version__)
except Exception as exc:
    print("  × lancedb 不可用：", exc)
for mod, label, hint in (("pypdf", "PDF 解析", "PDF 会被跳过"),
                         ("docx", "DOCX 解析", "DOCX 会被跳过"),
                         ("jieba", "中文分词", "全文索引退回 ngram(2-3) 字符切片"),
                         ("fastembed", "中文语义向量", "向量回落 hash，只能靠全文检索")):
    if mod == "jieba":
        try:
            import jieba
            print(f"  ✓ {label}（jieba {jieba.__version__}）")
        except Exception:
            print(f"  · {label} 不可用（缺 jieba：{hint}）")
        continue
    try:
        __import__(mod)
        print(f"  ✓ {label}（{mod}）")
    except Exception:
        print(f"  · {label} 不可用（缺 {mod}：{hint}）")
PY
}

if [ "${DEPS_ONLY:-0}" = 1 ]; then
    install_deps
    say ""
    say "依赖装好了。MCP 服务用 mcp/launch.sh 启动，它会自动找到 .venv。"
    exit 0
fi

if [ "$DEPS" = 1 ]; then
    install_deps
    say ""
else
    say "→ 跳过依赖安装（--no-deps）"
    say ""
fi

# --------------------------------------------------------------------------- #
# 2. 注册到本地市场
# --------------------------------------------------------------------------- #

LANCE_KB_DRY=$DRY "$PY" - "$SCRIPT_DIR" "$MARKET_ROOT" "$PLUGIN_NAME" "$MARKET_NAME" <<'PY'
import json, os, shutil, sys, time

plugin_dir, market_root, name, market = sys.argv[1:5]
dry = os.environ.get("LANCE_KB_DRY") == "1"
WB = os.path.join(os.path.expanduser("~"), ".workbuddy")
TS = time.strftime("%Y%m%d-%H%M%S")
fail = []


def rd(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as e:
        print(f"× {path}\n  不是合法 JSON（{e}），先修好再装")
        sys.exit(2)


def wr(path, data, label):
    if dry:
        print(f"[dry-run] {label} → {path}")
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(path):
        shutil.copy2(path, f"{path}.bak-{TS}")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    print(f"✓ {label}\n  {path}")


print("环境自检：")
if sys.version_info < (3, 9):
    fail.append(f"Python {sys.version.split()[0]} < 3.9")
    print(f"  × Python {sys.version.split()[0]} 过低，需要 ≥ 3.9")
else:
    print(f"  ✓ 注册用 Python {sys.version.split()[0]}")

venv_py = os.path.join(plugin_dir, ".venv", "bin", "python")
if os.path.exists(venv_py):
    print(f"  ✓ 插件虚拟环境 {venv_py}")
else:
    print(f"  · 没有 .venv，服务会去找系统里已装 lancedb 的解释器")

ver = "0.0.0"
desc = ""
try:
    with open(os.path.join(plugin_dir, ".codebuddy-plugin", "plugin.json"), encoding="utf-8") as f:
        meta = json.load(f)
    ver = meta.get("version", ver)
    desc = meta.get("description", "")
except Exception:
    pass
print(f"  · 插件版本 {ver}")

kb_home = os.path.join(plugin_dir, "data")
config = os.path.join(kb_home, "config.json")
if os.path.exists(config):
    try:
        with open(config, encoding="utf-8") as f:
            cfg = json.load(f)
        emb = cfg.get("embedder") or {}
        print(f"  ✓ 已有知识库：嵌入后端 {emb.get('name')}/{emb.get('model')} · {emb.get('dim')} 维")
    except Exception:
        print("  · 配置文件读不出来，首次使用会重建")
else:
    print("  · 知识库还是空的，装完在对话里让它索引一个目录即可")
print()

now = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

manifest = {
    "name": market,
    "description": "本地插件市场：插件直接用源码目录，改代码重启即生效，无需打包。",
    "plugins": [
        {
            "name": name,
            "source": f"./{name}",
            "version": ver,
            "description": desc or "本地知识库（LanceDB）",
        }
    ],
}
wr(os.path.join(market_root, ".codebuddy-plugin", "marketplace.json"), manifest, "市场清单")

km_path = os.path.join(WB, "plugins", "known_marketplaces.json")
km = rd(km_path, {})
km[market] = {
    "manifestName": market,
    "type": "directory",
    "source": {"source": "directory", "path": market_root},
    "installLocation": market_root,
    "description": f"Marketplace from {market_root}",
    "lastUpdated": now,
    "autoUpdate": False,
}
wr(km_path, km, "登记市场")

ip_path = os.path.join(WB, "plugins", "installed_plugins.json")
ip = rd(ip_path, {"version": 2, "plugins": {}})
ip.setdefault("version", 2)
ip.setdefault("plugins", {})
ip["plugins"][f"{name}@{market}"] = [
    {
        "scope": "user",
        "installPath": plugin_dir,
        "version": ver,
        "installedAt": now,
        "lastUpdated": now,
    }
]
wr(ip_path, ip, "登记安装")

st_path = os.path.join(WB, "settings.json")
st = rd(st_path, {})
en = st.setdefault("enabledPlugins", {})
en[f"{name}@{market}"] = True
wr(st_path, st, "启用插件")

mc_path = os.path.join(WB, "mcp.json")
mc = rd(mc_path, {"mcpServers": {}})
if name in mc.get("mcpServers", {}):
    del mc["mcpServers"][name]
    wr(mc_path, mc, "清理 mcp.json 里的同名服务")
else:
    print("✓ mcp.json 无同名服务，跳过")

print("\n" + ("自检（dry-run，读的是内存里的预期值，未落盘）：" if dry else "自检："))
print(f"  market    : {market in km}")
print(f"  installed : {f'{name}@{market}' in ip['plugins']}  {plugin_dir}")
print(f"  enabled   : {en.get(f'{name}@{market}')}")
print(f"  no dup    : {name not in mc.get('mcpServers', {})}")

if fail:
    print("\n× 有阻塞问题：" + "；".join(fail))
    sys.exit(3)
print("\n下一步：完全退出 WorkBuddy 再打开（enabledPlugins 只在启动时读），")
print("      首次调用会弹「信任此 MCP 服务」，点信任。")
PY

say ""
say "先跑一遍自检与试检索："
say "  $VENV/bin/python mcp/kb.py stats"
say "  $VENV/bin/python mcp/kb.py add ~/某个目录 --tags 测试"
say "  $VENV/bin/python mcp/kb.py search -q '随便搜点东西'"
