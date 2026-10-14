#!/usr/bin/env bash
# One-command engine bootstrap for Linux (Ubuntu 22.04 walkthrough:
# docs/DEPLOY_UBUNTU.md).
#
#   scripts/bootstrap-engine.sh                # both engines, build included
#   scripts/bootstrap-engine.sh -e kvmem      # mode B only (256K KVMem)
#   scripts/bootstrap-engine.sh -e llama      # mode A only (128K llama.cpp)
#   scripts/bootstrap-engine.sh --no-build    # clone + patch only
#
# Mode B: clones the KVMem tree, replays both published patches (via
# scripts/apply-engine-patches.sh, idempotent) and builds llama-kvmem-server
# with the flags of the upstream tree's own scripts/build-cuda.sh, except the
# CUDA architecture which defaults here to 75-real for Turing.
#
# Mode A: clones stock llama.cpp at 16378d9, applies the Turing routing patch
# and builds llama-server with the flags documented in docs/ENGINE_NOTES.md.
#
# Safe to re-run: checkouts, patches and the cmake build directory are all
# incremental.

set -euo pipefail

KV_LLAMA_PIN=16378d93f94012d4228c8c7683adce3f286aee5d
KV_UPSTREAM=https://github.com/kvmem/kvmem-llama.cpp
LLAMA_UPSTREAM=https://github.com/ggml-org/llama.cpp.git

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENGINE=all
JOBS="$(nproc 2>/dev/null || echo 8)"
CUDA_ARCH=75-real
DO_BUILD=1

usage() {
    sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        -e) ENGINE="${2:?}"; shift 2 ;;
        -j) JOBS="${2:?}"; shift 2 ;;
        --cuda-arch) CUDA_ARCH="${2:?}"; shift 2 ;;
        --no-build) DO_BUILD=0; shift ;;
        -h|--help) usage ;;
        *) echo "unknown option: $1" >&2; usage ;;
    esac
done
case "$ENGINE" in kvmem|llama|all) ;; *) echo "-e must be kvmem, llama or all" >&2; exit 2 ;; esac

command -v git >/dev/null 2>&1 || { echo 'missing tool: git (see docs/DEPLOY_UBUNTU.md)' >&2; exit 1; }
if [ "$DO_BUILD" -eq 1 ]; then
    for tool in cmake ninja; do
        command -v "$tool" >/dev/null 2>&1 || { echo "missing tool: $tool (see docs/DEPLOY_UBUNTU.md)" >&2; exit 1; }
    done
    # The NVIDIA apt packages put nvcc under /usr/local/cuda without touching PATH.
    if ! command -v nvcc >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nvcc ]; then
        export PATH="/usr/local/cuda/bin:$PATH"
        export CUDA_HOME=/usr/local/cuda
    fi
    command -v nvcc >/dev/null 2>&1 || { echo 'missing tool: nvcc (install a CUDA toolkit, see docs/DEPLOY_UBUNTU.md)' >&2; exit 1; }
    echo "using $(command -v nvcc) ($(nvcc --version | tail -1 | awk '{print $NF}'))"
fi

configure() {
    local source_dir="$1" build_dir="$2"; shift 2
    cmake -S "$source_dir" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        "$@"
}

apply_patch_idempotent() {
    local dir="$1" patch="$2" label="$3"
    if git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
        echo "      already applied: $label"
    else
        git -C "$dir" apply --check "$patch"
        git -C "$dir" apply "$patch"
        echo "      applied: $label"
    fi
}

build_kvmem() {
    local kv_dir="${KVMEM_DIR:-$REPO_ROOT/kvmem-llama.cpp}"

    echo '[1/2] KVMem tree: pinned checkouts + published patches'
    if [ ! -d "$kv_dir/.git" ]; then
        echo "      cloning $KV_UPSTREAM"
        git clone "$KV_UPSTREAM" "$kv_dir"
    fi
    KVMEM_DIR="$kv_dir" bash "$REPO_ROOT/scripts/apply-engine-patches.sh"

    if [ "$DO_BUILD" -eq 1 ]; then
        echo '[2/2] building llama-kvmem-server (CUDA '"$CUDA_ARCH"')'
        configure "$kv_dir" "$kv_dir/build-linux" \
            -DGGML_CUDA=ON \
            -DGGML_CUDA_FA_ALL_QUANTS=ON \
            -DKVMEM_BUILD_LLAMA=ON \
            -DLLAMA_KVMEM=ON \
            -DLLAMA_KVMEM_ROOT="$kv_dir" \
            -DLLAMA_CURL=OFF \
            -DLLAMA_BUILD_TESTS=OFF
        cmake --build "$kv_dir/build-linux" --target llama-kvmem-server -j"$JOBS"
        echo "      binary: $kv_dir/build-linux/bin/llama-kvmem-server"
        echo "      point config.sh at it (KVMemServer) and run scripts/start-kvmem.sh"
    fi
}

build_llama() {
    local llama_dir="${LLAMA_DIR:-$REPO_ROOT/llama.cpp}"
    local patch_turing="$REPO_ROOT/patches/turing-mmvq-mmq-routing.diff"

    echo '[1/2] llama.cpp tree: pin 16378d9 + Turing routing patch'
    if [ ! -d "$llama_dir/.git" ]; then
        echo "      cloning $LLAMA_UPSTREAM"
        git clone "$LLAMA_UPSTREAM" "$llama_dir"
    fi
    git -C "$llama_dir" cat-file -e "$KV_LLAMA_PIN^{commit}" 2>/dev/null || {
        echo "commit $KV_LLAMA_PIN not present; run 'git -C $llama_dir fetch' and retry" >&2; exit 1; }
    git -C "$llama_dir" -c advice.detachedHead=false checkout --quiet "$KV_LLAMA_PIN"
    apply_patch_idempotent "$llama_dir" "$patch_turing" 'turing-mmvq-mmq-routing.diff (1 file)'

    if [ "$DO_BUILD" -eq 1 ]; then
        echo '[2/2] building llama-server (CUDA '"$CUDA_ARCH"')'
        configure "$llama_dir" "$llama_dir/build-linux" \
            -DGGML_CUDA=ON \
            -DGGML_NATIVE=ON \
            -DGGML_CUDA_FA_ALL_QUANTS=ON \
            -DLLAMA_CURL=OFF \
            -DLLAMA_BUILD_TESTS=OFF \
            -DLLAMA_BUILD_EXAMPLES=ON
        cmake --build "$llama_dir/build-linux" --target llama-server -j"$JOBS"
        echo "      binary: $llama_dir/build-linux/bin/llama-server"
        echo "      point config.sh at it (LlamaServer) and run scripts/start-llama.sh"
    fi
}

case "$ENGINE" in
    kvmem) build_kvmem ;;
    llama) build_llama ;;
    all)   build_kvmem; build_llama ;;
esac
