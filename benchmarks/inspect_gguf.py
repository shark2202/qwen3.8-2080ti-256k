#!/usr/bin/env python3
"""GGUF 结构 + 量化分布核验。

回答的问题不是"文件完不完整"（那是 sha256 的事），而是：
  · 这到底是真 2-bit 档，还是换了个名字的高位宽档？→ 位宽直方图 + 有效 bpw
  · 有没有 MTP 头 / 视觉投影？→ 决定能不能开自投机、能不能吃图
  · 张量类型里有没有本机 sm_75 特别容易出问题的？→ 对照 RUNBOOK §8 的教训

用法：
  PYTHONPATH=<llama.cpp>/gguf-py python3 inspect_gguf.py a.gguf [b.gguf ...]
  python3 inspect_gguf.py --ref '{"IQ3_S":268,...}' a.gguf     # 与给定分布逐类型对比
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, "/home/a/ai-work/qwen3.8-2080ti-256k/kvmem-llama.cpp/llama.cpp/gguf-py")
from gguf import GGUFReader  # noqa: E402
from gguf.constants import GGMLQuantizationType  # noqa: E402

# 本机已有的对照：models/llm/Q2-LynnStyle/manifest.json 记录的同族量化
REF_Q2_LYNNSTYLE = {"Q5_K": 2, "F32": 353, "Q4_K": 85, "IQ3_S": 268,
                    "IQ3_XXS": 32, "IQ4_XS": 88, "IQ2_S": 21, "Q2_K": 2}


def meta(reader: GGUFReader, key: str, default=None):
    """跨 gguf-py 版本取元数据：优先 contents()，退回 value，最后 default。"""
    f = reader.fields.get(key)
    if f is None:
        return default
    for getter in (lambda: f.contents(), lambda: f.value):
        try:
            v = getter()
        except Exception:
            continue
        if v is None:
            continue
        if callable(v):
            try:
                v = v()
            except Exception:
                continue
        if isinstance(v, (list, tuple)):
            if not v:
                continue
            v = v[0]
        if hasattr(v, "item"):
            try:
                v = v.item()
            except Exception:
                pass
        if isinstance(v, bytes):
            try:
                v = v.decode()
            except Exception:
                pass
        return v
    return default


def inspect(path: str) -> dict:
    r = GGUFReader(path)
    arch = meta(r, "general.architecture", "?")
    hist: dict[str, int] = {}
    bytes_by_type: dict[str, int] = {}
    n_params = 0
    for t in r.tensors:
        name = GGMLQuantizationType(t.tensor_type).name if isinstance(t.tensor_type, int) else str(t.tensor_type)
        hist[name] = hist.get(name, 0) + 1
        bytes_by_type[name] = bytes_by_type.get(name, 0) + int(t.n_bytes)
        n_params += int(t.n_elements)
    total_bytes = sum(bytes_by_type.values())
    return {
        "path": path,
        "file_bytes": os.path.getsize(path),
        "arch": arch,
        "name": meta(r, "general.name"),
        "file_type": meta(r, "general.file_type"),
        "tensors": len(r.tensors),
        "params": n_params,
        "metrics": {k: meta(r, f"{arch}.{k}") for k in
                    ("block_count", "context_length", "embedding_length",
                     "nextn_predict_layers", "attention.head_count")},
        "vision": bool(meta(r, f"{arch}.vision.block_count")),
        "weight_bytes": total_bytes,
        "bpw": round(total_bytes * 8 / n_params, 3) if n_params else None,
        "hist": dict(sorted(hist.items(), key=lambda kv: -bytes_by_type[kv[0]])),
        "bytes_by_type": dict(sorted(bytes_by_type.items(), key=lambda kv: -kv[1])),
    }


def show(d: dict, ref: dict | None) -> None:
    print(f"\n=== {os.path.basename(d['path'])} ===")
    print(f"  文件 {d['file_bytes'] / 2**30:.3f} GiB | 权重 {d['weight_bytes'] / 2**30:.3f} GiB"
          f" | 张量 {d['tensors']} | 参数 {d['params'] / 1e9:.2f} B | 有效 {d['bpw']} bpw")
    m = d["metrics"]
    print(f"  架构 {d['arch']} | 层 {m.get('block_count')} | ctx {m.get('context_length')}"
          f" | nextn(MTP) {m.get('nextn_predict_layers')} | 视觉 {'有' if d['vision'] else '无'}")
    print("  量化分布（按占字节排序）：")
    for t, b in d["bytes_by_type"].items():
        share = b / d["weight_bytes"] * 100
        print(f"    {t:<10} {d['hist'][t]:>4} 个  {b / 2**30:>7.3f} GiB  {share:>5.1f}%")
    if ref:
        print("  与对照分布对比：")
        keys = sorted(set(ref) | set(d["hist"]))
        for k in keys:
            a, b = ref.get(k, 0), d["hist"].get(k, 0)
            if a or b:
                print(f"    {k:<10} 对照 {a:>4} → 本件 {b:>4}  {'' if a == b else f'({b - a:+d})'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ref", default=None, help="对照分布 JSON；默认给 Q2-LynnStyle")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    ref = json.loads(a.ref) if a.ref else REF_Q2_LYNNSTYLE
    out = []
    for p in a.files:
        try:
            d = inspect(p)
        except Exception as e:  # 文件不完整/不是 GGUF 时给出人话
            print(f"\n❌ {p}: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        out.append(d)
        if not a.json:
            show(d, ref if len(a.files) == 1 else None)
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())