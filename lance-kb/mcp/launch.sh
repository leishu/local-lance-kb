#!/bin/sh
# lance-kb MCP 启动器：找一个能 import lancedb 的解释器，然后 exec server.py。
#
# 为什么需要它：MCP 的 command 写死解释器路径，换台机器就失效；
# 写死 python3 又常常是系统 Python，没装 lancedb，服务会直接起不来。
#
# 约定：
#   * stdout 是 MCP 协议通道，本脚本任何情况下都不往 stdout 写非协议内容；
#     诊断信息一律走 stderr（--which 除外）。
#   * 探测只依赖 shell 内建，不依赖 dirname/ls/sort（客户端可能给很精简的 PATH）。
#   * 用 exec 替换进程，保证 stdin/stdout 与信号原样透传。

set -u

log() { printf '[lance-kb] %s\n' "$*" >&2; }

case "$0" in
    */*) HERE=$(cd "${0%/*}" && pwd) ;;
    *)   HERE=$(pwd) ;;
esac

candidates() {
    printf '%s\n' \
        "$HERE/../.venv/bin/python3" \
        "$HERE/../.venv/bin/python" \
        "$HOME/.workbuddy/binaries/python/envs/default/bin/python" \
        "$HOME/.workbuddy/binaries/python/envs/lance-kb/bin/python"

    for d in "$HOME"/.workbuddy/binaries/python/versions/*/; do
        [ -d "$d" ] || continue
        printf '%s\n' "${d}bin/python3" "${d}bin/python"
    done

    printf '%s\n' python3 python /usr/bin/python3 /usr/local/bin/python3
}

probe() {
    py=$1
    [ -n "$py" ] || return 1
    if [ -x "$py" ]; then
        :
    elif command -v "$py" >/dev/null 2>&1; then
        :
    else
        return 1
    fi
    "$py" - <<'PYCODE' >/dev/null 2>&1
import sys
if sys.version_info < (3, 9):
    sys.exit(1)
import lancedb, pyarrow  # noqa: F401
PYCODE
}

PY=""

if [ -n "${LANCE_KB_PYTHON:-}" ]; then
    if probe "$LANCE_KB_PYTHON"; then
        PY=$LANCE_KB_PYTHON
    else
        log "LANCE_KB_PYTHON 指定的解释器不可用：$LANCE_KB_PYTHON"
        log "需要 Python >= 3.9 且已安装 lancedb"
        exit 1
    fi
fi

if [ -z "$PY" ]; then
    while IFS= read -r c; do
        if probe "$c"; then
            PY=$c
            break
        fi
    done <<EOF
$(candidates)
EOF
fi

if [ -z "$PY" ]; then
    log "找不到装了 lancedb 的 Python 解释器。"
    log "在插件目录跑一次安装脚本，它会建 .venv 并装好依赖："
    log "  sh $HERE/../install.sh --deps-only"
    log "或用 LANCE_KB_PYTHON=/path/to/python3 显式指定。"
    exit 1
fi

if [ "${1:-}" = "--which" ]; then
    printf '%s\n' "$PY"
    exit 0
fi

log "python=$PY ($("$PY" -V 2>&1))"
exec "$PY" "$HERE/server.py" "$@"
