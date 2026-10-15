#!/usr/bin/env python3
"""Long-context + speed ladder against a running llama-server (mode B / KVMem).

Builds a real-prose haystack from the repo's own markdown, plants three
NEEDLE-X=NNNN records at 10/50/90% depth, then asks for all three at growing
prompt sizes. Prints prompt_n, cache_n, prefill/decode tok/s, needle hits and
free VRAM per step.

The chars->tokens ratio is calibrated with a cache-proof probe (unique nonce),
because a repeated prompt is served from the prefix cache and reports prompt_n=1.
Sizes are re-estimated after every successful step and a 400 ("prompt +
max_tokens exceeds n_ctx") shrinks the prompt by 10% and retries.
"""
import json, os, subprocess, time, urllib.error, urllib.request

BASE = "http://127.0.0.1:8080"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAY = os.environ.get("HAY", os.path.join(REPO, "benchmarks", "haystack.md"))
TARGETS = [32768, 131072, 245760]   # prompt tokens; 245760 + 600 max_tokens stays under n_ctx 262144
NEEDLES = [("A", "4172"), ("B", "8391"), ("C", "2604")]
QUESTION = (
    "\n\n---\n以上文档里埋有三条形如 NEEDLE-X=NNNN 的记录（X 为 A、B、C）。"
    "只输出三行，按 A、B、C 顺序给出它们的完整原文，不要解释。"
)

def haystack():
    """Use HAY when it exists, else repeat the repo's own markdown to ~2.6M chars."""
    if os.path.exists(HAY):
        return open(HAY, encoding="utf-8", errors="ignore").read()
    docs = []
    for root, dirs, names in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in (".git", "logs", "llama.cpp", "kvmem-llama.cpp")]
        docs += [open(os.path.join(root, n), encoding="utf-8", errors="ignore").read()
                 for n in sorted(names) if n.endswith(".md")]
    unit = "\n\n".join(docs)
    return unit * (2_600_000 // max(1, len(unit)) + 1)


hay = haystack()
print(f"haystack: {len(hay)} chars ({HAY})", flush=True)


def build(nchars):
    body = hay[:nchars]
    parts, last = [], 0
    for (tag, val), frac in zip(NEEDLES, (0.1, 0.5, 0.9)):
        p = int(len(body) * frac)
        parts.append(body[last:p])
        parts.append(f"\n\n[RECORD] 验证标记 NEEDLE-{tag}={val}\n\n")
        last = p
    parts.append(body[last:])
    return "".join(parts)


def vram_free():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip()
    return int(out)


def ask(text, max_tokens=600, label=""):
    payload = {
        "model": "qwen3.8",
        "messages": [{"role": "user", "content": text + QUESTION}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "reasoning_effort": "low",
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:200]
        print(f"[{label}] HTTP {e.code}: {body} (chars={len(text)})", flush=True)
        raise
    wall = time.time() - t0
    t = d["timings"]
    msg = d["choices"][0]["message"]
    ans = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    hits = [tag for tag, val in NEEDLES if val in ans]
    hits_r = [tag for tag, val in NEEDLES if val in reasoning]
    print(
        f"[{label}] chars={len(text)} prompt_n={t['prompt_n']} cache_n={t['cache_n']} "
        f"prefill={t['prompt_per_second']:.1f} tok/s decode={t['predicted_per_second']:.1f} tok/s "
        f"draft={t.get('draft_n_accepted','-')}/{t.get('draft_n','-')} "
        f"wall={wall:.0f}s needles={len(hits)}/3 think={len(hits_r)}/3 free_vram={vram_free()}MiB "
        f"finish={d['choices'][0]['finish_reason']}",
        flush=True,
    )
    print(f"    answer: {ans.strip()[:220]!r} reasoning_chars={len(reasoning)}", flush=True)
    return t, len(hits)


# --- cache-proof calibration: a unique nonce defeats the prefix cache ---
nonce = os.urandom(6).hex()
probe = f"<!-- nonce {nonce} -->\n" + build(4000)
t0, _ = ask(probe, max_tokens=8, label="calib")
ratio = len(probe + QUESTION) / max(1, t0["prompt_n"])
print(f"calibration: {ratio:.3f} chars/token", flush=True)

for target in TARGETS:
    if vram_free() < 800:
        print(f"ABORT before {target}: free VRAM below 800 MiB", flush=True)
        break
    nchars = min(int(target * ratio), len(hay))
    if int(target * ratio) > len(hay):
        print(f"note: haystack too small for {target} tokens; using the whole {len(hay)} chars", flush=True)
    for attempt in range(4):
        try:
            t, _ = ask(build(nchars), max_tokens=600, label=f"target~{target}")
            ratio = (nchars + len(QUESTION)) / max(1, t["prompt_n"])   # refine for the next step
            break
        except urllib.error.HTTPError:
            nchars = int(nchars * 0.9)
            print(f"    retry with chars={nchars}", flush=True)
    else:
        print(f"[target~{target}] gave up after 4 attempts", flush=True)
    print(f"    free_vram={vram_free()}MiB", flush=True)

print("done", flush=True)