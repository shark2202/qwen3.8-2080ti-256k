#!/usr/bin/env bash
# Download the model files both run modes need (walkthrough: docs/DEPLOY_UBUNTU.md
# §5; full file list and checksums: docs/MODELS.md).
#
#   scripts/download-models.sh                     # into ~/ai-work/models/llm, auto endpoint
#   scripts/download-models.sh -d /data/models     # elsewhere
#   scripts/download-models.sh --endpoint https://huggingface.co
#
# Safe to re-run: `hf download` skips files already complete and resumes
# partial (.incomplete) transfers. Checksums are verified at the end; the
# download host comes from --endpoint / $HF_ENDPOINT, defaulting to
# huggingface.co with a fallback to the hf-mirror.com mirror when the direct
# route is unreachable.

set -euo pipefail

MODELS_DIR="$HOME/ai-work/models/llm"
ENDPOINT="${HF_ENDPOINT:-}"
GGUF_REPO=nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2-GGUF
TEMPLATE_REPO=froggeric/Qwen-Fixed-Chat-Templates
TEMPLATE_SHA16=c47c82b0544752d4   # v22.4 pinned in docs/DEPLOY_UBUNTU.md §5

usage() {
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        -d) MODELS_DIR="${2:?}"; shift 2 ;;
        --endpoint) ENDPOINT="${2:?}"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "unknown option: $1" >&2; usage ;;
    esac
done

HF_BIN="${HF_BIN:-$(command -v hf || true)}"
if [ -z "$HF_BIN" ] && [ -x "$HOME/ai-work/.venv/bin/hf" ]; then
    HF_BIN="$HOME/ai-work/.venv/bin/hf"
fi
[ -n "$HF_BIN" ] || { echo 'missing tool: hf (pip install -U "huggingface_hub[cli]")' >&2; exit 1; }

if [ -z "$ENDPOINT" ]; then
    if curl -sS -m 6 -o /dev/null https://huggingface.co 2>/dev/null; then
        ENDPOINT=https://huggingface.co
    else
        ENDPOINT=https://hf-mirror.com
        echo "huggingface.co unreachable; using $ENDPOINT (--endpoint to override)" >&2
    fi
fi
export HF_ENDPOINT="$ENDPOINT"
case "$ENDPOINT" in
    https://huggingface.co) ;;
    *) export HF_HUB_DISABLE_XET=1   # third-party mirrors do not speak the xet protocol
esac

"$HF_BIN" download "$GGUF_REPO" --include "Q2-LynnStyle/*" --local-dir "$MODELS_DIR"
"$HF_BIN" download "$TEMPLATE_REPO" chat_template.jinja --local-dir "$MODELS_DIR"

(cd "$MODELS_DIR/Q2-LynnStyle" && sha256sum --ignore-missing -c SHA256SUMS)

got16="$(sha256sum "$MODELS_DIR/chat_template.jinja" | cut -c1-16)"
if [ "$got16" != "$TEMPLATE_SHA16" ]; then
    echo "chat_template.jinja sha starts $got16, pinned docs use $TEMPLATE_SHA16:" \
         "upstream moved on — re-check reasoning-effort behaviour (docs/DEPLOY_UBUNTU.md §5)" >&2
else
    echo "chat_template.jinja OK ($got16)"
fi
