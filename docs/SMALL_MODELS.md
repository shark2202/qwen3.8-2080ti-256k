# 在这套配置上跑小参数模型（本地笔记）

> 本机工作笔记，中文，**随仓库发布**（`README_ZH.md` 同例）；待办：翻译成英文并与
> `docs/MODELS.md`、`docs/LAUNCH_PARAMS.md` 合并。已跟踪的启动/检查脚本与
> `minimax-h3-turing` 手册都引用本文件，所以先收进仓库，避免悬空引用。

## 结论

引擎（`llama-server` / `llama-kvmem-server`）对模型本身无偏好，任意 GGUF 都能加载；
22 GB 显存跑 0.6B–14B 反而更宽裕，KV 更小，长上下文不再是瓶颈。

但两个启动脚本是按 27B + DFlash2 草稿写死的，直接换 `TargetModel` 会失败或变慢，
需要动下面四处。→ **2026-10-18 已改好并实测**（见本文末「实测 / 工作流 A / 工作流 B」）。

## 需要改的地方

| 位置 | 现状 | 换小模型时 |
|---|---|---|
| `config.sh` `TargetModel` | 27B Q2 GGUF | 改成小模型 GGUF 路径 |
| `config.sh` `DraftModel`；`start-llama.sh` 里 `--spec-draft-model` | DFlash2 草稿是为 27B 训练的；脚本还**硬校验 DraftModel 文件存在**，缺文件直接退出 | 草稿换成不匹配的只会被大量拒绝 → 更慢。设 `LlamaSpecType=none`（关投机）或 `ngram-mod`（无需草稿模型），并把脚本里的 `--spec-draft-*` 改成沿 `mmproj_args` 那种条件数组写法，同时把 `DraftModel` 移出必需文件循环 |
| `ChatTemplate` + `--jinja --reasoning on --reasoning-effort low` | froggeric 的 Qwen3 模板，`reasoning_effort` 由它实现 | 非 Qwen 系列要换模板，或去掉这几个参数 |
| `GGML_MMVQ_ALL=1` / `MmvqMax=4` | 脚本注释：为这套 IQ3_S 权重集所需，"在 IQ4_XS 为主的权重上反而掉约 9%" | 小模型多为 Q4_K_M / Q8_0 全 K-quant → `MmvqAll=0` |

`GGML_MMVQ_MAX` 那条"必须小于验证宽度（draft 深度 + 1）"的约束只在开投机时成立。

**已实测**：本机 `llama.cpp/build-linux/bin/llama-server --help` 的 `--spec-type` 合法值为
`none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache`
—— 即 `none` 与 `ngram-mod` 都可用，关草稿不需要改引擎。

## 怎么用（2026-10-18 已落地）

启动脚本现在按 `LlamaSpecType` 里有没有 `draft-` 来决定要不要草稿：没有 `draft-*` 时
**不校验 `DraftModel` 文件、也不传 `--spec-draft-*`**（`scripts/start-llama.sh`、
`scripts/start-kvmem.sh` 都改了）。顺带：`CONFIG_PATH` 可覆盖配置文件，`--alias` 由
`ModelAlias` 决定，`check-config.sh` 会把草稿显示成 `absent (optional)` 并给一行说明。

所以小模型**不用动 `config.sh`**，用一份独立配置即可（本机已写好 `config.small.sh`）：

```bash
# config.small.sh 的四处差异：
#   TargetModel     9B heretic Q4_K_M
#   LlamaSpecType   ngram-mod     无草稿（none 也行）
#   MmvqAll         0             Q4_K_M 权重集，不是 IQ3_S 那套
#   LlamaContext    65536         写剧本用不到 128K
CONFIG_PATH=config.small.sh scripts/check-config.sh
CONFIG_PATH=config.small.sh scripts/start-llama.sh
```

验证"确实没传草稿参数"（应为 0）：`tr '\0' '\n' < /proc/<llama-server-pid>/cmdline | grep -c spec-draft`。

## 本机模型位置（2026-10-17 实测）

语言模型（本仓库用的 GGUF）——`/home/a/ai-work/models/llm/`：

| 文件 | 大小 | 角色 |
|---|---:|---|
| `Q2-LynnStyle/Qwen3.8-27B-EfficientThink-SimPO-Q2-LynnStyle.gguf` | 13.0 GB | 目标模型（`config.sh` `TargetModel`） |
| `Q2-LynnStyle/dflash2-qwen38-27b-Q4_K_M.gguf` | 1.14 GB | 草稿模型（`DraftModel`） |
| `Q2-LynnStyle/dflash2-qwen38-27b-Q8_0.gguf` | 2.06 GB | 草稿备选 |
| `Q2-LynnStyle/mmproj-Qwen3.8-27B-Q4_K_M.gguf` | 522 MB | 视觉投影（`MmprojModel`） |
| `Q2-LynnStyle/mtp-Qwen3.8-27B-{Q4_0,Q8_0}.gguf` | 1.68 / 3.16 GB | MTP 头；当前 Q2 目标无 `nextn` 头，不可用 |
| `chat_template.jinja` | 27 KB | Qwen3 聊天模板（`ChatTemplate`） |

图像 / 视频模型（ComfyUI + minimax-h3-turing）——`/home/a/ai-work/models/`，
经 `ComfyUI/extra_model_paths.yaml` 的 `h3_w4a8_weights` 挂载：

| 文件 | 大小 |
|---|---:|
| `diffusion_models/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | 12.5 GB |
| `text_encoders/qwen3vl_4b_fp8_scaled.safetensors` | 5.2 GB |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 5.2 GB |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 605 MB |
| `loras/MiniMax-H3-{FL2VA,Ref2VA}-Acc-8Step_comfy.safetensors` | 1.72 GB ×2 |
| `loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_*.safetensors` | 1.96 GB |
| `loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_*.safetensors` | 315 MB |

**2026-10-18：小模型已下载并跑通**，`/home/a/ai-work/models/llm/Qwen3.8-9B-heretic/`：

| 文件 | 大小 | 出处 |
|---|---:|---|
| `Qwen3.8-9B-heretic-uncensored.Q4_K_M.gguf` | 5.24 GiB | `mradermacher/Qwen3.8-9B-heretic-uncensored-GGUF`（`rohit267` 的 heretic 去审查版） |

- 字节数 `5629109728`，sha256 `8eb51ad276c3b529266b800a2bd866ba03337709db6a1b4758243ffc5c96510f`（装前脚本验过，落位后独立复算一致）
- GGUF 头 `general.architecture = qwen35` → 本机引擎认识（`llama-arch.cpp` 里有 `qwen35`）
- **拉取/校验统一走清单**：`~/ai-work/minimax-h3-turing/models/MANIFEST.tsv`（18 个模型 / 59.53 GiB 的唯一记录）
  + `scripts/fetch_models.sh`（断点续传、装前字节数+sha256 双校验）：`--check` 校验、`--only 9B-heretic` 只补这个；
  旧的一次性脚本 `~/ai-work/minimax-h3-turing/models/ranged_get.sh` 仍在。hf-mirror 直链 302 到 xet，聚合约 2 MB/s
- 其它候选（未下载）：`Qwen3.8-4B-Heretic.i1-GGUF`（2.5 GiB，最小）、`Qwen3.8-9B-Distill-SLERP-F451-Pro-Writer-Uncensored-GGUF`（写作特化）、`empero-ai/Qwen3.8-9B-Distill-GGUF`（未去审查的基座）。磁盘余量 `/` ≈740 GB。

## 实测（2026-10-18，RTX 2080 Ti 22G）

| 项 | 结果 |
|---|---|
| 加载 | arch `qwen35` 正常，`n_ctx_slot = 65536`，启动参数里 `spec-draft` 计数 **0** |
| 显存 | **6808 MiB**（权重 + 64K q8_0 KV）→ 余 ~15 GB，**装不下 H3，两套必须错峰** |
| 速度 | **70-71 tok/s**（冷启首答 9.7s / 684 tokens，`reasoning low`） |
| 无审查 | 要"雨夜持刀、血迹、写实不回避"的镜头脚本，直接给可拍内容，无劝诫、无免责声明 |
| 剧本 | `screenplay.py` 一次出 5 镜：**1084 tokens / 15.5s / 70 tok/s**，契约 0 违例（单行英文三段式、124 帧网格、跨镜主体/服装/色调一致） |

## 工作流 A：命令行写剧本 → 出片

```bash
CONFIG_PATH=config.small.sh scripts/start-llama.sh
python3 scripts/screenplay.py --brief "竹林剑客，黑色电影，复仇前夜；冷调晨雾，手持镜头" \
    --shots 5 --seed 3013 --out ~/ai-work/minimax-h3-turing/h3_demo_output/screenplay_bamboo
# -> screenplay_bamboo.md（人看） + screenplay_bamboo.shots.json（出片用）

# 出片前腾显存（22G 装不下两套）。注意 [Q] 写法：直接写全名 pkill 会匹配到自己
pkill -f "[Q]wen3.8-9B-heretic-uncensored"
cd ~/ai-work/minimax-h3-turing
P=$(python3 -c "import json;print(json.load(open('h3_demo_output/screenplay_bamboo.shots.json'))['shots'][0]['prompt_en'])")
scripts/h3turing.local.sh demo --prompt "$P" --seed 3013 --out ./h3_demo_output
```

提示词契约在 `scripts/screenplay.system.md`（要改风格就改它，不用碰代码）；
`--thinking none` 可省掉思考 token，`--seed` 让同一版剧本可复现。

## 工作流 B：ComfyUI 里一步到底（同一次排队：写剧本 → 出片）

- 节点源码：`ComfyUI/custom_nodes/ComfyUI-QwenScript/`，UI 里叫 **Qwen3.8 Script (local LLM)**
  （分类 MiniMax H3；输出 `prompt` 与 `screenplay`，节点上直接显示整份剧本）
- 工作流：`minimax-h3-turing/workflows/h3_w4a8_t2v_qwenscript_api.json`
  = t2v 成片档 + 该节点，`[6].prompt` 由字面量改成接到节点的 `prompt` 输出
- 22 GB 不能同时装 LLM 和 H3，所以**节点自己管生命周期**：
  1. 先让 ComfyUI 卸载自己的模型缓存（紧随其后的就是采样器，安全）
  2. 端点没人应答就按 `CONFIG_PATH` 拉起 llama-server，有人应答就复用
  3. 生成剧本 → **只杀掉自己拉起的那个进程**（按 `TargetModel` 路径匹配，不误杀）
  4. 采样器拿到整卡
- `reuse_script=true` + `script_out=<前缀>`：同一份 `.shots.json` 后续排队直接复用，
  改 `shot` 序号即可逐镜出片，不必反复问模型（模型再快也不如不跑）
- **显存卸载的机制**（三层：进程内丢引用 → `torch.cuda.empty_cache()` 把块还给驱动 → 跨进程只能杀）与 22 GB 账本，
  见 `~/ai-work/minimax-h3-turing/RUNBOOK.local.md` §1.2；节点源码目录的 README 也有一份精简版

## 仍未验证

- 模式 B（KVMem）配小模型没跑过；`patches/` 里 KVMem 补丁对架构参数的假设没核对。
- 无审查只测了暴力/黑色这一个题材域；情色等其它域未测。`thinking=none` 与 `low` 的出片质量差异也没对比。
- 镜头数 >12 时 JSON 契约的稳定性未测（失败时节点/脚本会把模型原始回复留成 `<script_out>.raw.txt`，便于查契约问题）。
- 上游发布：本文件是本机笔记，并入 `docs/MODELS.md` / `docs/LAUNCH_PARAMS.md` 需要英文翻译与合并。