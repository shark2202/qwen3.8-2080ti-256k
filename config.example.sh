# Linux launcher configuration. Copy this file to config.sh and edit the paths
# (and only the paths, unless you are deliberately re-tuning -- every value is
# explained in docs/LAUNCH_PARAMS.md). Keep model weights outside the repository.
#
# Sourced by scripts/start-llama.sh, scripts/start-kvmem.sh and
# scripts/check-config.sh. Ubuntu deployment walkthrough: docs/DEPLOY_UBUNTU.md.

# ---- shared ----
TargetModel='/data/models/Q2-LynnStyle/Qwen3.8-27B-EfficientThink-SimPO-Q2-LynnStyle.gguf'
DraftModel='/data/models/Q2-LynnStyle/dflash2-qwen38-27b-Q4_K_M.gguf'
MmprojModel='/data/models/Q2-LynnStyle/mmproj-Qwen3.8-27B-Q4_K_M.gguf'   # optional; kept in host RAM
ChatTemplate='/data/models/chat_template.jinja'                          # see docs/MODELS.md
#LogDir=''          # optional; defaults to <repo>/logs
Host=127.0.0.1      # bind address; 0.0.0.0 serves the whole LAN (no auth unless ApiKey is set)
#ApiKey='sk-local'  # optional; when set, clients send Authorization: Bearer <key>
Port=8080
Threads=8           # host threads; match the physical cores of the machine

# ---- mode A: upstream llama.cpp + DFlash2, 128K (scripts/start-llama.sh) ----
LlamaServer="$HOME/engines/llama.cpp/build-linux/bin/llama-server"
LlamaContext=131072
LlamaKVType=q8_0
LlamaSpecType=draft-dflash,ngram-mod
LlamaDraftDepth=6
LlamaBatchSize=1024
LlamaUBatchSize=512
MmvqMax=4   # must stay below the verification width (draft depth + 1)
MmvqAll=1   # required for this IQ3_S-heavy weight set

# ---- mode B: KVMem, 256K logical (scripts/start-kvmem.sh) ----
KVMemServer="$HOME/engines/kvmem-llama.cpp/build-linux/bin/llama-kvmem-server"
ContextSize=262144
KVMemBudget=96256
GenerationReserve=16384
KVMemBlockTokens=128
KVDtype=q8_0
DraftDepth=6
SpecType=draft-dflash,ngram-mod
BatchSize=1024
UBatchSize=512
