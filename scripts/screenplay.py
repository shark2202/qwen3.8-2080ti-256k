#!/usr/bin/env python3
"""Write a shot-by-shot 剧本 for the H3 video model with the local small model.

Talks to the OpenAI-compatible endpoint of scripts/start-llama.sh (default
http://127.0.0.1:8080/v1) -- stdlib only, no SDK. The prompt contract lives in
scripts/screenplay.system.md, next to this file, so it can be edited without
touching code.

    python3 scripts/screenplay.py --brief "竹林剑客，黑色电影，复仇" --shots 5 \
        --out "$HOME/ai-work/h3_demo_output/screenplay_bamboo"

Writes <out>.md (human readable) and <out>.shots.json (machine readable, the
one to feed into h3turing demo --prompt). Prints the measured wall time and
tokens/s, because those are the numbers the launch params are tuned against.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def post_json(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_json(text: str) -> dict:
    """The model is asked for bare JSON; tolerate a fenced block or prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.split("\n", 1)[1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start:end + 1])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--brief", required=True, help="题材/梗概/要求，越具体越好")
    p.add_argument("--shots", type=int, default=5, help="镜头数（默认 5）")
    p.add_argument("--base", default="http://127.0.0.1:8080/v1",
                   help="OpenAI-compatible base URL (default: local mode A server)")
    p.add_argument("--model", default="qwen3.8-9b-heretic",
                   help="model alias, as printed by GET /v1/models")
    p.add_argument("--system", default=str(HERE / "screenplay.system.md"))
    p.add_argument("--out", default="screenplay", help="output prefix, .md/.shots.json appended")
    p.add_argument("--thinking", default="low", choices=["none", "low", "medium", "high"],
                   help="reasoning effort; the chat template implements these levels")
    p.add_argument("--temperature", type=float, default=0.8, help="剧本要发散，比默认 0.7 略高")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--seed", type=int, default=0, help="采样种子；同 prompt 同 seed 出同一版剧本")
    a = p.parse_args()

    system = Path(a.system).read_text(encoding="utf-8")
    user = f"{a.brief}\n\n镜头数：{a.shots}。"
    payload = {
        "model": a.model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": a.temperature,
        "seed": a.seed,
        "response_format": {"type": "json_object"},
    }
    if a.thinking != "medium":  # medium is the template's default
        # 必须走 chat_template_kwargs：OpenAI 的 reasoning_effort 字段本引擎不映射（实测 no-op），
        # kwargs 才真正传进模板；实测 none 关思考后 60.7 tok/s，low 时 41.5
        payload["chat_template_kwargs"] = {"reasoning_effort": a.thinking}

    t0 = time.time()
    try:
        data = post_json(a.base.rstrip("/") + "/chat/completions", payload, a.timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:400]
        print(f"error: HTTP {e.code} from {a.base}: {body}")
        return 1
    except urllib.error.URLError as e:
        print(f"error: cannot reach {a.base} ({e.reason}). "
              "Start it with CONFIG_PATH=config.small.sh scripts/start-llama.sh")
        return 1
    wall = time.time() - t0

    choice = data["choices"][0]["message"]
    usage = data.get("usage") or {}
    out_tokens = usage.get("completion_tokens")
    print(f"{data.get('model')}: {out_tokens} tokens in {wall:.1f}s "
          f"({out_tokens / wall if out_tokens else 0:.1f} tok/s)")

    text = choice.get("content") or ""
    try:
        script = extract_json(text)
        shots = script["shots"]
        assert shots, "empty shot list"
    except Exception as e:
        # Keep the raw reply: a JSON failure is a prompt/parse problem to inspect,
        # not something to silently retry.
        raw = Path(a.out + ".raw.txt")
        raw.write_text(text or str(choice), encoding="utf-8")
        print(f"error: reply is not usable JSON ({e}); raw text kept at {raw}")
        return 1

    md = [f"# {script.get('title', '(untitled)')}", "",
          f"**logline**: {script.get('logline', '')}", "",
          f"_由 {data.get('model')} 生成，{wall:.0f}s / {out_tokens or '?'} tokens_", ""]
    for i, s in enumerate(shots, 1):
        md += [f"## {s.get('index', i)}. {s.get('title', '')}  ({s.get('frames', 124)} 帧)", "",
               f"- 画面：{s.get('scene_zh', '')}",
               f"- 首帧：{s.get('first_frame_hint') or '不需要（文生视频）'}", "",
               "```text", s.get("prompt_en", ""), "```", ""]
    Path(a.out + ".md").write_text("\n".join(md), encoding="utf-8")
    Path(a.out + ".shots.json").write_text(
        json.dumps({"model": data.get("model"), "brief": a.brief, "shots": shots},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"剧本: {a.out}.md  ({len(shots)} 镜)")
    print(f"出片用: {a.out}.shots.json  ->  "
          f"python3 -c \"import json,sys;print(json.load(open(sys.argv[1]))['shots'][0]['prompt_en'])\" {a.out}.shots.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())