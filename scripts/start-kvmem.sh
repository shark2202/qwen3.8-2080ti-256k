#!/usr/bin/env bash
# Mode B launcher -- KVMem, 256K logical context, port 8080.
# Linux counterpart of scripts/start-kvmem.ps1 (same arguments, same defaults).
#
#   engine  llama-kvmem-server built from the KVMem tree + the two patches in
#           patches/; scripts/bootstrap-engine.sh builds it. The --kvmem-*
#           flags below do not exist in stock upstream llama.cpp.
#   memory  112,640-token physical GPU working set (96,256 retrieval budget +
#           16,384 generation reserve); older history is retrieved, not resident.
#           One generation cannot exceed GenerationReserve tokens, thinking
#           included -- raising it requires taking the space out of KVMemBudget.
#
# This script owns llama-kvmem-server only. Mode A (llama.cpp) is a different
# binary on the same port; the two are mutually exclusive on a 22 GB card.
# Keep >= 800 MiB VRAM free (see docs/LAUNCH_PARAMS.md section 3). Linux
# launchers are for Linux only (they use /proc and ss).

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
: "${ContextSize:=262144}"
: "${KVMemBudget:=96256}"
: "${GenerationReserve:=16384}"
: "${KVMemBlockTokens:=128}"
: "${KVDtype:=q8_0}"
: "${DraftDepth:=6}"
: "${SpecType:=draft-dflash,ngram-mod}"
: "${BatchSize:=1024}"
: "${UBatchSize:=512}"
: "${MmvqMax:=4}"
: "${MmvqAll:=1}"
: "${Port:=8080}"
: "${Threads:=8}"
: "${LogDir:=$REPO_ROOT/logs}"

fail=0
for path in "$KVMemServer" "$TargetModel" "$DraftModel" "$ChatTemplate"; do
    if [ -z "${path:-}" ] || [ ! -f "$path" ]; then
        echo "Missing required file: ${path:-<unset>}" >&2
        fail=1
    fi
done
if [ "$fail" -ne 0 ]; then
    exit 1
fi

# Stop an instance of THIS server only, never somebody else's process.
server_abs="$(readlink -f "$KVMemServer")"
for proc in /proc/[0-9]*; do
    exe="$(readlink "$proc/exe" 2>/dev/null || true)"
    if [ "$exe" = "$server_abs" ]; then
        kill "${proc##*/}"
        echo "Stopped this configured llama-kvmem-server (pid ${proc##*/})."
        exit 0
    fi
done

# Mode A shares the port. Warn rather than killing whatever holds it.
if command -v ss >/dev/null 2>&1; then
    listener="$(ss -ltnp "sport = :$Port" 2>/dev/null | awk 'NR>1 && $1=="LISTEN" {print; exit}')"
    if [ -n "$listener" ]; then
        echo "Port $Port is already in use: $listener" >&2
        echo 'Stop it before starting KVMem.' >&2
        exit 1
    fi
fi

mkdir -p "$LogDir"
timestamp="$(date +%Y%m%d-%H%M%S)"
stdout="$LogDir/kvmem-$timestamp.out.log"
stderr="$LogDir/kvmem-$timestamp.err.log"

# Environment of the server process. These are read once at start.
export KVMEM_LAZY_DECODE_MEAN=1
# KVMEM_RESUME_INTERVAL=2048 and KVMEM_TRACE are documented in
# docs/LAUNCH_PARAMS.md section 2.3; the Windows launcher the published
# numbers were taken with sets neither, so neither does this one.
# Turing MMVQ->MMQ routing, same per-weight-set caveat as mode A:
# GGML_MMVQ_ALL=1 is required for this IQ3_S-heavy weight set.
export GGML_MMVQ_MAX="$MmvqMax"
export GGML_MMVQ_ALL="$MmvqAll"

mmproj_args=()
if [ -n "${MmprojModel:-}" ] && [ -f "$MmprojModel" ]; then
    mmproj_args=(--mmproj "$MmprojModel" --no-mmproj-offload)
fi

echo "Starting mode B: KVMem 256K logical (working set $((KVMemBudget + GenerationReserve)) tokens, spec=$SpecType depth=$DraftDepth MMVQ_MAX=$MmvqMax ALL=$MmvqAll)"
nohup "$KVMemServer" \
    --model "$TargetModel" \
    --alias qwen3.8-local \
    --host 127.0.0.1 --port "$Port" \
    -c "$ContextSize" \
    -n "$GenerationReserve" \
    -ngl 99 \
    --kvmem \
    --kvmem-method retrieval \
    --kvmem-budget "$KVMemBudget" \
    --kvmem-gen-reserve "$GenerationReserve" \
    --kvmem-block-tokens "$KVMemBlockTokens" \
    --kv-dtype "$KVDtype" \
    --spec-type "$SpecType" \
    --spec-draft-model "$DraftModel" \
    --spec-draft-n-max "$DraftDepth" \
    --spec-kv-dtype f16 \
    --chat-template-file "$ChatTemplate" \
    --reasoning-effort low \
    --enable-thinking \
    --temp 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --min-p 0.05 \
    -b "$BatchSize" \
    -ub "$UBatchSize" \
    --threads "$Threads" \
    --threads-batch "$Threads" \
    "${mmproj_args[@]}" \
    >"$stdout" 2>"$stderr" &
echo "API:  http://127.0.0.1:$Port  (pid $!)"
echo "Logs: $stderr"
