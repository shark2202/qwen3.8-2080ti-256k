# Deploying on Ubuntu 22.04 (single RTX 2080 Ti 22 GB)

This is the Linux counterpart of the Windows quick start in the README: same
two run modes, same model files, same every-value-explained reference in
[LAUNCH_PARAMS.md](LAUNCH_PARAMS.md). Linux assets:

| Windows | Linux | Purpose |
|---|---|---|
| `config.example.ps1` | `config.example.sh` | copy to `config.sh`, edit paths |
| `scripts\bootstrap-engine.ps1 -Build` | `scripts/bootstrap-engine.sh` | clone upstreams, replay patches, build |
| `scripts\check-config.ps1` | `scripts/check-config.sh` | validate `config.sh` |
| `scripts\start-llama.ps1` | `scripts/start-llama.sh` | mode A: 128K llama.cpp + DFlash2 |
| `scripts\start-kvmem.ps1` | `scripts/start-kvmem.sh` | mode B: 256K logical, KVMem |

The launchers are Linux-only (they match processes via `/proc/*/exe` and check
the port with `ss`); the start scripts produced byte-identical argument lists
to the Windows ones when tested against a stub server.

## 1. GPU driver

Turing, so any current proprietary driver works. Ubuntu 22.04 ships 535 in its
own archive; 550/560 via the PPA are fine too:

```bash
sudo ubuntu-drivers install nvidia:560     # or: sudo apt install nvidia-driver-560
sudo reboot
nvidia-smi                                 # must list the GPU with ~22528 MiB
```

If `nvidia-smi` shows 11 GB on a modded 22 GB card, the driver is fine but the
card's strap was flashed for 11 GB — fix that before anything else; the mode B
working set does not fit in 11 GB.

## 2. Build dependencies

```bash
sudo apt update
sudo apt install -y build-essential git cmake ninja-build ccache pkg-config \
                    python3-pip
```

## 3. CUDA toolkit

The recipe was measured with CUDA 13.4 on Windows; on Linux any CUDA 12.4+ or
13.x toolkit builds it (Turing `sm_75` is retained in CUDA 13; Maxwell/Pascal/
Volta are the families that were dropped). Stock gcc-11 on Ubuntu 22.04 is
accepted by both 12.x and 13.x.

From the NVIDIA network repo (keys and pin as per
<https://developer.nvidia.com/cuda-downloads>, then):

```bash
sudo apt install -y cuda-toolkit-12-6     # driver >= 560; or cuda-toolkit-13-x with driver >= 580
```

The apt packages install nvcc under `/usr/local/cuda-12.6/bin` without touching
`PATH`. Either add it yourself:

```bash
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
```

…or do nothing: `scripts/bootstrap-engine.sh` auto-detects `/usr/local/cuda/bin/nvcc`.
Verify with `nvcc --version`.

## 4. Engines (one command)

```bash
git clone https://github.com/shark2202/qwen3.8-2080ti-256k
cd qwen3.8-2080ti-256k
git checkout dev                       # while the Linux assets live on the dev branch
scripts/bootstrap-engine.sh            # both engines; -e kvmem / -e llama to pick one
```

What it does, per engine:

- **Mode B (KVMem)** — clones `kvmem/kvmem-llama.cpp`, then
  `scripts/apply-engine-patches.sh` checks out both pinned commits and replays
  `patches/kvmem-outer-local-changes.patch` (10 files) and
  `patches/llama-kvmem-current.patch` (40 files); then cmake builds the
  `llama-kvmem-server` target with the upstream tree's own `build-cuda.sh`
  flags — `GGML_CUDA=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON` (required for `q8_0` KV
  with flash attention), `KVMEM_BUILD_LLAMA`/`LLAMA_KVMEM` — with
  `-DCMAKE_CUDA_ARCHITECTURES=75-real`. Result:
  `kvmem-llama.cpp/build-linux/bin/llama-kvmem-server`.
- **Mode A (llama.cpp)** — clones `ggml-org/llama.cpp`, checks out `16378d9`,
  applies `patches/turing-mmvq-mmq-routing.diff` (without it the published
  speeds are unreachable) and builds `llama-server` with the flags from
  [ENGINE_NOTES.md](ENGINE_NOTES.md). Result:
  `llama.cpp/build-linux/bin/llama-server`.

`75-real` (not `75`) avoids shipping PTX the card would have to JIT at load.
Everything is incremental: re-running the script re-uses the checkouts, skips
already-applied patches and re-runs only the cmake build.

First build of both engines takes roughly 15–40 minutes depending on CPU.

## 5. Model files

```bash
pip install -U "huggingface_hub[cli]"
hf download \
  nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2-GGUF \
  --include "Q2-LynnStyle/*" --local-dir ~/ai-work/models/llm
cd ~/ai-work/models/llm/Q2-LynnStyle && sha256sum --ignore-missing -c SHA256SUMS
```

Also download the chat template — it is what implements `reasoning_effort`:

```bash
hf download froggeric/Qwen-Fixed-Chat-Templates chat_template.jinja --local-dir ~/ai-work/models/llm
sha256sum ~/ai-work/models/llm/chat_template.jinja   # expect c47c82b0544752d4... (v22.4)
```

If the hash differs, upstream has moved on (the file was at v22.5 at the time
of writing); either pin the older revision with
`--revision <commit from the file's history>` or use the newer template and
re-check the reasoning-effort behaviour. File list, sizes and full checksums:
[MODELS.md](MODELS.md).

You need at least `Qwen3.8-27B-EfficientThink-SimPO-Q2-LynnStyle.gguf` (12.11 GiB),
`dflash2-qwen38-27b-Q4_K_M.gguf` (1.06 GiB) and the template; `mmproj-…-Q4_K_M.gguf`
is optional (image input; kept in host RAM either way). ~14 GiB of disk for the
minimum set.

## 6. Configure

```bash
cp config.example.sh config.sh
editor config.sh     # point the five paths at your files; Threads = physical cores
scripts/check-config.sh
```

## 7. Run

```bash
scripts/start-kvmem.sh    # mode B: 256K logical context  (the headline deployment)
scripts/start-llama.sh    # mode A: 128K daily driver      (alternative, same port)
```

Run the same script again to stop that server; the scripts never kill a
process they did not start and refuse the port if someone else holds it. Logs
land in `logs/` with timestamps (the interesting output is the `.err.log` —
llama.cpp logs to stderr).

First launch takes a while (12 GiB of weights to load). Then:

```bash
curl -s http://127.0.0.1:8080/v1/models | head
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-local","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":32}'
```

The OpenAI-compatible chat-completions API (plus `/v1/models` and `/health`) is
served on `127.0.0.1:8080`; there is no native Anthropic-format route, so front
it with a translating proxy if your client speaks that.

## 8. Verify the VRAM budget

```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

Mode B should sit near 21 GiB used with **≥ 800 MiB free** (the same floor as
Windows: below ~700 MiB the CUDA caching allocator falls back to
cudaMalloc/cudaFree churn; Windows additionally pages via WDDM, Linux does not
— but the allocator floor applies on both). After any context/quant change,
re-read the headroom before trusting the configuration. If it is tight, drop
`--cache-type-k/-v` to `q4_0` (mode A) or shrink `KVMemBudget` (mode B) before
touching weights or context.

## 9. Linux-specific notes

- **No WDDM**: Linux does not page VRAM to host RAM, so the failure mode of a
  too-tight budget is allocator churn + slowdowns rather than silent paging.
  The 800 MiB floor still applies.
- **Benchmarking**: the Windows "measure at 60 Hz" compositor caveat has no
  direct equivalent; on a headless box just keep the GPU otherwise idle. The
  cold-prompt rule (never re-use one prompt when measuring speculative
  decoding — the n-gram table survives across requests) applies unchanged,
  see [LAUNCH_PARAMS.md](LAUNCH_PARAMS.md) §4.
- **Autostart**: the launchers are interactive stop/start toggles, so prefer
  them over systemd units; a unit would fight the toggle. If you want mode B
  at boot regardless, wrap `scripts/start-kvmem.sh` in a `Type=oneshot`
  `RemainAfterExit=yes` unit and stop it via the same script.
- **`Threads`** in `config.sh` should match physical cores of the machine
  (host-side tokenization/sampling only — everything heavy is on the GPU).

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `patch does not apply` | The checkouts moved off their pins — re-run `scripts/apply-engine-patches.sh` (it re-checks out first), or a CRLF-mangled clone: this repo pins LF in `.gitattributes`; do not set `core.autocrlf=true` globally on Linux. |
| `nvcc not found` | Toolkit not installed or not on `PATH` — see §3; the bootstrap auto-detects `/usr/local/cuda/bin/nvcc`. |
| `Unsupported gpu architecture '75'` | Old CMake (< 3.18) or wrong toolkit; Ubuntu 22.04's `cmake` from the archive is fine. |
| gcc rejected by nvcc | You exported a too-new gcc; CUDA 12.6 accepts up to gcc 13 — the stock gcc-11 needs no override. |
| OOM / CUDA error at load | Check `nvidia-smi` for leftover processes (a mode A server still holding the card), then headroom §8. |
| Server starts, then exits | `logs/kvmem-*.err.log` has the reason; a missing chat template or a mismatched drafter GGUF are the usual ones. |
