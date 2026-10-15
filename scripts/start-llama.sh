#!/usr/bin/env bash
# Mode A launcher -- upstream llama.cpp + DFlash2, 128K context, port 8080.
# Linux counterpart of scripts/start-llama.ps1 (same arguments, same defaults).
#
#   engine  llama-server built from upstream llama.cpp 16378d9 with the Turing
#           MMVQ->MMQ routing patch (patches/turing-mmvq-mmq-routing.diff);
#           scripts/bootstrap-engine.sh -e llama builds it. Without the patch
#           the speculative verification batches fall back to MMVQ and the
#           published speeds are not reachable; see docs/LAUNCH_PARAMS.md section 4.
#   model   Q2-LynnStyle target GGUF (text-only, no welded MTP head)
#   draft   DFlash2 Q4_K_M block-diffusion drafter, speculative depth 6
#   spec    --spec-type draft-dflash,ngram-mod
#           ngram runs FIRST (priority is hardcoded in the engine: ngram* >
#           draft*), fires only on genuinely repeated content and abstains
#           otherwise, so on free-flowing text this is identical to
#           draft-dflash alone. Revert with LlamaSpecType=draft-dflash.
#   KV      q8_0 / q8_0 at 128K. 128K is the largest verified context for this
#           model; beyond ~130K tokens it is reported to emit EOS early
#           (llama.cpp #27756). Use scripts/start-kvmem.sh for 256K.
#
# This script owns llama-server only. Mode B (KVMem) is a different binary on
# the same port; the two are mutually exclusive on a 22 GB card.
#
# VRAM headroom is load-bearing: below roughly 700 MiB free the CUDA caching
# allocator starts cudaMalloc/cudaFree churn (on Windows, WDDM additionally
# pages VRAM to host RAM; measured there: prefill 375 -> 93 t/s). Keep >= 800
# MiB free. Linux launchers are for Linux only (they use /proc and ss).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="$REPO_ROOT/config.sh"
if [ ! -f "$CONFIG_PATH" ]; then
    echo 'Missing config.sh. Copy config.example.sh to config.sh first.' >&2
    exit 1
fi
# shellcheck source=/dev/null
. "$CONFIG_PATH"

# ---- defaults for anything config.sh leaves undefined ----
: "${LlamaContext:=131072}"
: "${LlamaKVType:=q8_0}"
: "${LlamaSpecType:=draft-dflash,ngram-mod}"
: "${LlamaDraftDepth:=6}"
: "${LlamaBatchSize:=1024}"
: "${LlamaUBatchSize:=512}"
: "${MmvqMax:=4}"
: "${MmvqAll:=1}"
: "${Host:=127.0.0.1}"
: "${Port:=8080}"
: "${Threads:=8}"
: "${LogDir:=$REPO_ROOT/logs}"

fail=0
for path in "$LlamaServer" "$TargetModel" "$DraftModel" "$ChatTemplate"; do
    if [ -z "${path:-}" ] || [ ! -f "$path" ]; then
        echo "Missing required file: ${path:-<unset>}" >&2
        fail=1
    fi
done
if [ "$fail" -ne 0 ]; then
    echo 'Run scripts/check-config.sh for the full list, and see docs/MODELS.md.' >&2
    exit 1
fi

# Stop an instance of THIS server only, never somebody else's process.
server_abs="$(readlink -f "$LlamaServer")"
for proc in /proc/[0-9]*; do
    exe="$(readlink "$proc/exe" 2>/dev/null || true)"
    if [ "$exe" = "$server_abs" ]; then
        kill "${proc##*/}"
        echo "Stopped this configured llama-server (pid ${proc##*/})."
        exit 0
    fi
done

# Mode B shares the port. Warn rather than killing whatever holds it.
if command -v ss >/dev/null 2>&1; then
    listener="$(ss -ltnp "sport = :$Port" 2>/dev/null | awk 'NR>1 && $1=="LISTEN" {print; exit}')"
    if [ -n "$listener" ]; then
        echo "Port $Port is already in use: $listener" >&2
        echo 'Stop it before starting mode A.' >&2
        exit 1
    fi
fi

mkdir -p "$LogDir"
timestamp="$(date +%Y%m%d-%H%M%S)"
stdout="$LogDir/server-$timestamp.out.log"
stderr="$LogDir/server-$timestamp.err.log"

# Read once at process start by the patched ggml_cuda_should_use_mmvq():
#   GGML_MMVQ_MAX  routing threshold; must stay below the verification width
#                  (draft depth + 1) or the patch silently stops firing.
#   GGML_MMVQ_ALL  extend the threshold to every quant type MMQ supports.
#                  Required for this IQ3_S-heavy weight set; costs ~9% on an
#                  IQ4_XS-dominated one.
export GGML_MMVQ_MAX="$MmvqMax"
export GGML_MMVQ_ALL="$MmvqAll"

# The vision projector is kept in HOST memory: --mmproj-offload defaults to
# enabled and overrides --mmproj-device none, and pushing the 522 MB projector
# onto the GPU costs ~1.5 GiB (measured 21908 vs 20375 MiB used). Image input
# still works without the offload.
mmproj_args=()
if [ -n "${MmprojModel:-}" ] && [ -f "$MmprojModel" ]; then
    mmproj_args=(--mmproj "$MmprojModel" --mmproj-device none --no-mmproj-offload)
fi

auth_args=()
if [ -n "${ApiKey:-}" ]; then
    auth_args=(--api-key "$ApiKey")
fi

echo "Starting mode A: llama.cpp + DFlash2 (ctx=$LlamaContext KV=$LlamaKVType spec=$LlamaSpecType depth=$LlamaDraftDepth MMVQ_MAX=$MmvqMax ALL=$MmvqAll)"
nohup "$LlamaServer" \
    --model "$TargetModel" \
    --alias qwen3.8-local \
    --host "$Host" --port "$Port" \
    "${auth_args[@]}" \
    --ctx-size "$LlamaContext" \
    --parallel 1 \
    --n-gpu-layers all \
    --fit off \
    --flash-attn on \
    --cache-type-k "$LlamaKVType" \
    --cache-type-v "$LlamaKVType" \
    --spec-type "$LlamaSpecType" \
    --spec-draft-model "$DraftModel" \
    --spec-draft-n-max "$LlamaDraftDepth" \
    --spec-draft-type-k f16 \
    --spec-draft-type-v f16 \
    --jinja \
    --chat-template-file "$ChatTemplate" \
    --reasoning on \
    --reasoning-effort low \
    --temp 0.7 \
    --top-k 20 \
    --top-p 0.8 \
    --batch-size "$LlamaBatchSize" \
    --ubatch-size "$LlamaUBatchSize" \
    --threads "$Threads" \
    --threads-batch "$Threads" \
    "${mmproj_args[@]}" \
    >"$stdout" 2>"$stderr" &
echo "API:  http://${Host}:$Port  (pid $!)"
if [ "$Host" = "0.0.0.0" ]; then
    echo "      LAN:  http://$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}'):$Port/v1  (open to the whole subnet)"
fi
echo "Logs: $stderr"
