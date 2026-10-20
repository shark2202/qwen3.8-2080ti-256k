# Saluki 27B IQ2-mix 评估（候选 target · 2026-10-20 实测）

> 本机笔记，中文，与 [SMALL_MODELS.md](SMALL_MODELS.md) 同例。
> 被测：`ConwayResearch/Underdog-Saluki-27B-1.0` —— Qwen3.8-27B 的 ~2bit 混合量化 GGUF（Apache-2.0）。
> 命题：它能否顶替现役 `Q2-LynnStyle` 当 27B 目标？
> **结论：能。省 4.2 GiB 显存、慢 3%；换过去必须重调 `--spec-draft-n-max`。**
> 工具见 [`../benchmarks/`](../benchmarks/)（`inspect_gguf.py` / `run_suite.py` / `report.py`），原始报告 `benchmarks/*.json`。

## 1. 权重与校验

| 文件 | 角色 | 字节 | sha256 |
|---|---|---:|---|
| `Underdog-Saluki-27B-1.0-IQ2-mix.gguf` | 目标（text-only，无 MTP 头） | 7,898,369,152 | `4a673518b11b1c4445f9b9a9d3c40356f5ba39f6a6260dcb25f3fbfd475d9efb` |
| `Underdog-Saluki-27B-1.0-IQ2-mix-MTP.gguf` | 目标 + 嫁接 Qwen 自带 MTP 头（`Kujira/…-MTP-GGUF`） | 8,349,689,920 | `98f6ebb527917a2541df8e667d51c3ec8a8bb135b44c606d40cc6ed924e89e52` |

- HF 官方直连本机不通；**ModelScope 有同一份**（`Sha256` 字段与 HF 的 `x-linked-etag` 独立给出同一个值），实测 2.05 MB/s，比 hf-mirror 的 1.82 MB/s 快。
- MTP 那份 ModelScope 没有，只能走 hf-mirror。
- 取件用 `minimax-h3-turing/scripts/fetch_models.sh` 或 `saluki-test/fetch_one.sh` 那套单流续传；**不要用 `curl -C -` 配 `-r`**（见 RUNBOOK §4.6）。

## 2. 结构核验（`inspect_gguf.py`）

```
主件  7.356 GiB / 权重 7.346 GiB / 851 张量 / 26.90 B 参数 / 有效 2.346 bpw
      qwen35 · 64 层 · ctx 262144 · nextn(MTP) 无 · 视觉 无
MTP   7.776 GiB / 866 张量 / 27.32 B 参数 / 2.442 bpw · 65 层 · nextn = 1
```

量化分布（主件，按占字节）：`IQ2_XXS` 30.1% · `IQ3_S` 15.8% · `Q2_K` 11.3% · `IQ2_S` 11.0% · `IQ2_XS` 7.9% ·
`IQ1_M` 7.9% · `IQ3_XXS` 6.8% · **`IQ1_S` 6.4%** · `IQ4_XS` 1.6% · `Q4_K` 0.7% · `Q8_0` 0.3% · `F32` 0.1%

**两条对照结论**

- 与现役 `Q2-LynnStyle` 比：**3.863 bpw → 2.346 bpw**，真·1.65 倍压缩。那个 "Q2" 是档位名，实测 `IQ2_S` 只占 4.6% 字节（32% 是 `IQ3_S`、22% 是 `IQ4_XS`）；Saluki 才是真 2-bit 档（含 14.3% 的 `IQ1_*`）。
- 厂商说的 **"under 8 GB" 是文件体积**（7.356 GiB），不是显存 —— 32K + `q8_0` KV 下实占 **8928 MiB = 8.72 GiB**，8 GB 卡差 736 MiB。

**技巧**：张量表在文件最前面几百 KB，所以**不用等下完**就能读结构 —— 取前缀 64 MB + `truncate` 成稀疏文件（真实占用 64 MB），gguf-py 即可解析。本轮就是靠它在下完前就报了 2.346 bpw（事后与完整文件逐字段一致）。

## 3. 行为断言（`run_suite.py`，7 条固定题面）

**7/7 全过**：`PONG` / `402` / 中文通顺 / 单 tool call / **并行 2 个 tool call** / thinking 答对 `$0.05`。

⚠️ 这一层不是形式：**2.346 bpw 的 `IQ1`/`IQ2` 混合在本机 sm_75 上没有任何静默劣化**。对照先例 —— `int8-convrot`（走 comfy-kitchen 专用内核）在 Turing 上是"`status: success` + 全黑 mp4"。**换任何低位宽权重都先跑这层**。

## 4. 三条路线实测（32K · `q8_0` KV · 7 条题面 · decode 取 `decode_n ≥ 40` 的中位）

| 配置 | 权重 | 显存 | 净增 | decode tok/s | 断言 |
|---|---:|---:|---:|---:|---|
| `Q2-LynnStyle` 无投机（对照） | 12.11 GiB | 13224 MiB | — | 22.5 | ✅ |
| `Q2-LynnStyle` + DFlash2（对照） | 12.11 | 15806 | +3409 | **65.7** | ✅ |
| Saluki 无投机 | 7.36 | 8928 | — | 23.2 | ✅ |
| Saluki + **DFlash2**（现成草稿，零新增下载） | 7.36 | 11510 | **+2582** | **63.7** | ✅ |
| Saluki + **MTP** `n-max 2` | 7.78 | 9922 | **+994** | 41.9 | ✅ |
| Saluki + **MTP** `n-max 4` | 7.78 | 10222 | **+1294** | 51.8 | ✅ |

### 4.1 DFlash2 胜 MTP 23%，但 MTP 省 1.3 GiB —— 差在**接受长度**，不是接受率

服务端日志的 `draft acceptance`（接受率 / 每次验证接受的长度）：

```
DFlash2(0.75–0.92)  mean len 5.50 / 6.11 / 6.50
MTP n4 (0.74–1.00)  mean len 2.66 / 3.98 / 5.00
MTP n2 (0.62–1.00)  mean len 2.24 / 2.66 / 3.00
```

DFlash2 是 block diffusion，**一次前向出一整块**；MTP 是一 token 一 token 拼块 —— 所以 MTP 的接受率能到 1.00 却依然更慢。
**22 GB 卡有空间 → 用 DFlash2；8–12 GB 卡放不下草稿时才轮到 MTP。**

### 4.2 `GGML_MMVQ_ALL` 只影响投机路径（已写回 LAUNCH_PARAMS §1.3）

```
无投机：ALL=0 → 23.2    ALL=1 → 23.2   （完全一样）
有投机：ALL=0 → 49.9    ALL=1 → 63.7   （+28%）
```

"`IQ4_XS` 为主时 ALL=1 掉 9%"那条**不能外推到低位宽档** —— 这个 `IQ1`/`IQ2` 为主的 GGUF 同样偏好 `1`。

### 4.3 MTP 的 "拐点在 n2" 没有迁移过来

同一 target 上 `n-max 4` 比 `n-max 2` 快 24%（51.8 vs 41.9），接受长度随深度增长（2.24 → 5.00）。
那条结论来自另一个 target/drafter 组合，**换草稿或换 target 都要重扫**。

## 5. 结论与代价

```
现役 Q2-LynnStyle (3.863 bpw) + DFlash2   15806 MiB   65.7 tok/s
Saluki           (2.346 bpw) + DFlash2    11510 MiB   63.7 tok/s
                                         ─ 4296 MiB    慢 3%
```

- **可以顶替**：省 4.2 GiB 显存换 3% 速度，直接扩大与 ComfyUI 抢 22 GB 时的余量。
- **仍不足以与 H3 同场**：H3 峰值 18926 + 11510 = 30436 ≫ 22528（见 RUNBOOK.local.md §1.2/§1.3），共存结论不变。
- **换过去必做**：重扫 `--spec-draft-n-max`；`MMVQ_ALL=1`；把草稿 KV 保持 `f16`。

## 6. 局限 / 未验证

- 每个变体**只跑一次**（无重复测量）；题面只有 7 条、中位数取 n=3–4；`dec 3/4` 的短输出测的是首 token 不是速度。
- 采样用 harness 的 `0.0/0.6`，不是生产 `0.7`；两个 target 的题面输出长度不同，逐步对比带噪声。
- **没测质量**：只验了"不坏"（sanity / 连贯 / tool call 格式）。厂商自称 tool-calling 保留 96%、竞赛数学掉到 82–85%，本仓库**没有可比的真实任务集**，所以这条既没证实也没推翻。
- 没测长上下文 / 重复内容下 `ngram-mod` 的贡献；没测 128K/256K；没测视觉附件（主件 text-only）。
- 厂商分数全部是自测、未独立复现，且**没有公开完整压缩配方**。