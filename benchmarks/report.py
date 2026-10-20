#!/usr/bin/env python3
"""把 run_suite.py 产出的 *.json 汇总成一张对比表。

    python3 report.py                       # 表 + 断言汇总
    python3 report.py --diff a b            # 再逐题并排两个 label 的原文

口径纪律：
  · decode 只统计 decode_n >= 40 的题面。n=3/4 的短输出测的是首 token，不是 decode 速度。
  · 显存用服务进程自身的 MiB（整卡含别人），附"权重外的开销"= 进程显存 - 权重字节。
  · 断言不过的行会标 ❌ —— 那一行的数字全部作废（先例：LTX int8-convrot 静默全黑）。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

MIN_DECODE_N = 40


def load(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        with open(p) as f:
            out.append(json.load(f))
    return out


def w_gib(r: dict) -> float:
    return r["file_gib"]


def overhead(r: dict) -> int:
    return r["vram_server_mib"] - int(r["file_gib"] * 1024**3 / 1024**2)


def decode_med(r: dict) -> tuple[float, int]:
    vals = [x["decode_tps"] for x in r["results"] if x.get("decode_n", 0) >= MIN_DECODE_N]
    return (statistics.median(vals) if vals else 0.0, len(vals))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/a/ai-work/saluki-test")
    ap.add_argument("--diff", nargs=2, metavar=("LABEL_A", "LABEL_B"))
    a = ap.parse_args()

    reports = load(sorted(glob.glob(os.path.join(a.dir, "*.json"))))
    if not reports:
        print("没有报告", file=sys.stderr)
        return 1

    if a.diff:
        ra = next((r for r in reports if r["label"] == a.diff[0]), None)
        rb = next((r for r in reports if r["label"] == a.diff[1]), None)
        if not ra or not rb:
            print(f"找不到 label：{[r['label'] for r in reports]}", file=sys.stderr)
            return 1
        print(f"=== {ra['label']}  ←→  {rb['label']} ===\n")
        for ca, cb in zip(ra["results"], rb["results"]):
            print(f"--- {ca['id']} ---")
            print(f"  [A] {ca['decode_tps']} tok/s(dec {ca.get('decode_n')})  "
                  f"calls={ca.get('n_tool_calls')}  {(ca.get('content') or '')[:300]!r}")
            if ca.get("tool_calls"):
                print(f"      args: {ca['tool_calls']}")
            print(f"  [B] {cb['decode_tps']} tok/s(dec {cb.get('decode_n')})  "
                  f"calls={cb.get('n_tool_calls')}  {(cb.get('content') or '')[:300]!r}")
            if cb.get("tool_calls"):
                print(f"      args: {cb['tool_calls']}")
            print()
        return 0

    hdr = f"{'label':<24}{'权重GiB':>9}{'显存MiB':>9}{'开销MiB':>9}{'ctx':>8}{'投机':>7}{'decode tok/s':>14}{'n':>4}  断言"
    print(hdr)
    print("-" * len(hdr))
    for r in reports:
        med, n = decode_med(r)
        spec = ("MTP" if (r.get("spec_mtp") or "draft-mtp" in r.get("extra", ""))
                else ("dflash" if "draft-dflash" in r.get("extra", "") else "-"))
        fail = r["assertion_failures"]
        mark = "✅" if not fail else "❌ " + ",".join(fail)
        print(f"{r['label']:<24}{w_gib(r):>9.2f}{r['vram_server_mib']:>9}{overhead(r):>9}"
              f"{r['ctx']:>8}{spec:>7}{med:>14.1f}{n:>4}  {mark}")

    print(f"\n（decode 中位数只取 decode_n >= {MIN_DECODE_N} 的题面；短输出测的是首 token 不是速度）")
    print("\n逐题断言：")
    for r in reports:
        print(f"  {r['label']:<24}", end="")
        print(" ".join(f"{c['id']}:{'✓' if c.get('ok') else ('·' if c.get('ok') is None else '✗')}"
                       for c in r["results"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())