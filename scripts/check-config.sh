#!/usr/bin/env bash
# Validate config.sh: required files present, tuning values consistent.
# Linux counterpart of scripts/check-config.ps1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config.sh}"
if [ ! -f "$CONFIG_PATH" ]; then
    echo 'Copy config.example.sh to config.sh first.' >&2
    exit 1
fi
# shellcheck source=/dev/null
. "$CONFIG_PATH"

# A DraftModel is required only when a spec source names a draft-* proposal --
# ngram-mod / none run without one (small models, docs/SMALL_MODELS.md).
spec_needs_draft=0
case ",${LlamaSpecType:-},${SpecType:-}," in
    *draft-*) spec_needs_draft=1 ;;
esac

show_path() {
    local label="$1" path="${2:-}" optional="$3"
    if [ -n "$path" ] && [ -e "$path" ]; then
        printf '%-22s %-8s %s\n' "$label" 'OK' "$path"
    elif [ "$optional" = 1 ]; then
        printf '%-22s %-8s %s\n' "$label" 'absent' '(optional)'
    else
        printf '%-22s %-8s %s\n' "$label" 'MISSING' "${path:-<unset>}"
    fi
}

echo '--- shared files ---'
show_path target_model  "${TargetModel:-}"  0
show_path draft_model   "${DraftModel:-}"   $([ "$spec_needs_draft" = 1 ] && echo 0 || echo 1)
show_path mmproj        "${MmprojModel:-}"  1
show_path chat_template "${ChatTemplate:-}" 0

echo
echo '--- mode A: llama.cpp + DFlash2 (128K) ---'
show_path llama-server "${LlamaServer:-}" 0
if [ -n "${LlamaContext:-}" ]; then
    kv_type="${LlamaKVType:-q8_0}"
    spec_type="${LlamaSpecType:-draft-dflash,ngram-mod}"
    draft_depth="${LlamaDraftDepth:-6}"
    printf 'context %s tokens, KV %s/%s, spec %s depth %s\n' \
        "$LlamaContext" "$kv_type" "$kv_type" "$spec_type" "$draft_depth"
    mmvq_max="${MmvqMax:-4}"
    mmvq_all="${MmvqAll:-1}"
    echo "MMVQ routing: GGML_MMVQ_MAX=$mmvq_max GGML_MMVQ_ALL=$mmvq_all"
    width=$((draft_depth + 1))
    if [ "$mmvq_max" -ge "$width" ] && [ "$spec_needs_draft" = 1 ]; then
        echo "  WARNING: MMVQ_MAX ($mmvq_max) >= verification width ($width): the routing patch will not fire."
    fi
fi
    if [ "$spec_needs_draft" = 0 ]; then
        echo '  note: spec has no draft-* source, so DraftModel is not required (docs/SMALL_MODELS.md)'
    fi

echo
echo '--- mode B: KVMem (256K logical) ---'
show_path llama-kvmem-server "${KVMemServer:-}" 0
budget="${KVMemBudget:-96256}"
reserve="${GenerationReserve:-16384}"
logical="${ContextSize:-262144}"
printf 'logical context    : %s tokens\n' "$logical"
printf 'GPU KV working set : %s tokens (budget %s + generation reserve %s)\n' \
    "$((budget + reserve))" "$budget" "$reserve"
printf 'speculation        : %s, DFlash depth %s\n' \
    "${SpecType:-draft-dflash,ngram-mod}" "${DraftDepth:-6}"

echo
echo 'Both modes bind the same port and are mutually exclusive; run one at a time.'
echo 'Every value above is explained in docs/LAUNCH_PARAMS.md.'
