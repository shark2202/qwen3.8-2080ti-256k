#!/usr/bin/env python3
"""Saluki 27B 实测：加载 / 显存 / 产物断言 / tool calling / 速度。

纯标准库，不依赖第三方包。一次调用完成"起服务 → 量显存 → 跑固定题面 → 收工"，
所以每次测量都在同一台机、同一时刻、同一套题面上，结果可对比。

    python3 run_suite.py --model <gguf> --label saluki [--spec-mtp] [--ctx 32768]

产物断言（对应 minimax-h3-turing/RUNBOOK §5.4 的第 3 层）：
    sanity_echo 必须恰好等于 PONG，sanity_math 必须恰好等于 402。
    这两条不过 = 反量化/模板链路有问题，后面所有数字都无意义（先例：LTX int8-convrot
    在 sm_75 上静默产出全黑 mp4）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

BIN = "/home/a/ai-work/qwen3.8-2080ti-256k/llama.cpp/build-linux/bin/llama-server"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
            "required": ["city"],
        },
    },
}
TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_local_time",
        "description": "Get the current local time for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}

# 题面固定：改题面 = 换一次实验，不要在同一批对比里改
SUITE = [
    # (id, 期望, messages, tools, thinking, max_tokens)
    ("sanity_echo", "PONG", [{"role": "user", "content": "Reply with the single word PONG and nothing else."}], None, False, 64),
    ("sanity_math", "402", [{"role": "user", "content": "What is 17*23+11? Answer with just the number, no words."}], None, False, 64),
    ("sanity_zh", None, [{"role": "user", "content": "用一句中文说明：为什么清晨的竹林会有雾？"}], None, False, 200),
    ("coherence_en", None, [{"role": "user", "content": "Write exactly one sentence describing a swordsman in a bamboo forest at dawn."}], None, False, 160),
    ("tool_single", "tool:get_weather", [{"role": "user", "content": "What's the weather in Chengdu in Celsius?"}], [WEATHER_TOOL], False, 256),
    ("tool_parallel", "2 calls", [{"role": "user", "content": "Get me the weather in Chengdu and the local time in Tokyo."}], [WEATHER_TOOL, TIME_TOOL], False, 320),
    ("thinking_on", None, [{"role": "user", "content": "A bat and a ball cost 1.10 total. The bat costs 1.00 more than the ball. How much is the ball?"}], None, True, 700),
]


def check(expect, content: str, calls: list) -> bool | None:
    """题面断言：None = 只记录不判定。expect 用 'tool:<name>' 或 'N calls' 表达工具期望。"""
    if not expect:
        return None
    if expect.endswith(" calls"):
        return len(calls) == int(expect.split()[0])
    if expect.startswith("tool:"):
        return expect[5:] in [c["function"]["name"] for c in calls]
    return content.strip().strip(".").strip() == expect

def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def gpu_used_mib() -> int:
    out = sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    return int(out.splitlines()[0]) if out else -1


def proc_mib(pid: int) -> int:
    out = sh(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
    for line in out.splitlines():
        p, _, m = line.partition(",")
        if p.strip() == str(pid):
            return int(m.strip())
    return 0


def post(base: str, path: str, payload: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get(base: str, path: str, timeout: float = 5) -> dict:
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.load(r)


def start_server(model: str, ctx: int, port: int, spec_mtp: bool, jinja: bool, mmvq_all: str, extra: str = "") -> tuple[subprocess.Popen, float]:
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(BIN), GGML_MMVQ_MAX="4", GGML_MMVQ_ALL=mmvq_all)
    args = [BIN, "-m", model, "--host", "127.0.0.1", "--port", str(port),
            "-ngl", "99", "-fa", "on", "-c", str(ctx), "-np", "1",
            "-ctk", "q8_0", "-ctv", "q8_0", "--threads", "28"]
    if jinja:
        args.append("--jinja")
    if spec_mtp:
        args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    if extra:
        args += extra.split()
    log = open(f"/home/a/ai-work/saluki-test/{os.path.basename(model)}.{port}.log", "w")
    t0 = time.time()
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    deadline = t0 + 420
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"服务提前退出（exit {proc.returncode}），看日志：{log.name}")
        try:
            if get(base, "/health", 2).get("status") == "ok":
                return proc, time.time() - t0
        except Exception:
            time.sleep(1.0)
    raise SystemExit(f"420s 内没就绪：{log.name}")


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), 15)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), 9)


def run_one(base: str, case, model: str) -> dict:
    cid, expect, messages, tools, thinking, max_tok = case
    payload = {
        "model": model, "messages": messages, "max_tokens": max_tok,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        "temperature": 0.0 if not thinking else 0.6,
        "top_p": 0.95 if thinking else 1.0, "top_k": 20,
    }
    if tools:
        payload["tools"] = tools
    t0 = time.time()
    try:
        r = post(base, "/v1/chat/completions", payload)
    except urllib.error.HTTPError as e:
        return {"id": cid, "error": f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"}
    dt = time.time() - t0
    msg = r["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    calls = msg.get("tool_calls") or []
    usage = r.get("usage") or {}
    tm = r.get("timings") or {}
    return {
        "id": cid, "secs": round(dt, 2),
        "content": content, "reasoning_chars": len(reasoning),
        "tool_calls": [{"name": c["function"]["name"], "args": c["function"]["arguments"]} for c in calls],
        "n_tool_calls": len(calls),
        "completion_tokens": usage.get("completion_tokens"),
        "decode_tps": round(tm.get("predicted_per_second", 0), 1),
        "decode_n": tm.get("predicted_n"), "prompt_n": tm.get("prompt_n"),
        "prefill_ms": tm.get("prompt_ms"),
        "expect": expect,
        "ok": check(expect, content, calls),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--spec-mtp", action="store_true")
    ap.add_argument("--jinja", action="store_true", default=True)
    ap.add_argument("--mmvq-all", default="0", help="IQ3_S 为主的权重集需要 1（见 SMALL_MODELS.md）")
    ap.add_argument("--extra", default="", help="追加给 llama-server 的原始参数，空格分隔")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.makedirs("/home/a/ai-work/saluki-test", exist_ok=True)
    gpu_before = gpu_used_mib()
    proc, load_s = start_server(a.model, a.ctx, a.port, a.spec_mtp, a.jinja, a.mmvq_all, a.extra)
    base = f"http://127.0.0.1:{a.port}"
    time.sleep(3)
    vram = proc_mib(proc.pid)
    gpu_after = gpu_used_mib()
    big = os.path.getsize(a.model) / 1024**3

    # 预热一次（不计入），让 KV/图缓存就位
    post(base, "/v1/chat/completions", {"model": a.label, "messages": [{"role": "user", "content": "hi"}],
                                        "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False}})

    results = [run_one(base, c, a.label) for c in SUITE]
    props = {}
    try:
        p = get(base, "/props")
        props = {k: p.get(k) for k in ("model_path", "n_ctx", "build_info", "chat_template") if k in p}
        props.pop("chat_template", None)
    except Exception:
        pass
    stop_server(proc)

    failed = [r["id"] for r in results if r.get("ok") is False]
    report = {
        "label": a.label, "model": os.path.basename(a.model), "file_gib": round(big, 3),
        "ctx": a.ctx, "spec_mtp": a.spec_mtp, "mmvq_all": a.mmvq_all, "extra": a.extra,
        "load_seconds": round(load_s, 1),
        "vram_gpu_used_before": gpu_before, "vram_server_mib": vram, "vram_gpu_used_after": gpu_after,
        "assertion_failures": failed, "props": props, "results": results,
    }
    out = a.out or f"/home/a/ai-work/saluki-test/{a.label}.json"
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n=== {a.label} · {report['model']} ===")
    print(f"文件 {report['file_gib']} GiB | 加载 {report['load_seconds']}s | ctx {a.ctx} | MTP {a.spec_mtp}")
    print(f"显存：服务进程 {vram} MiB（整卡 {gpu_before} → {gpu_after} MiB）")
    print(f"断言：{'✅ 全过' if not failed else '❌ 未过: ' + ','.join(failed)}")
    for r in results:
        nt = f" tool_calls={r['n_tool_calls']}" if r.get("n_tool_calls") else ""
        tt = f" {r['decode_tps']:>5} tok/s(dec {r['decode_n']})" if r.get("decode_tps") else ""
        print(f"  {r['id']:<15} {r['secs']:>6}s{tt}{nt}  {(r.get('content') or r.get('error') or '')[:90]!r}")
    print(f"→ {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
