#!/usr/bin/env python3
"""
attn_conformance.py — the attention cell convention, in two runtimes.

Node runs quilt-attention's OWN code, unmodified. Python runs pyattn.py, written from the
specification and sharing no source. The question is whether attention -- the arithmetic
that actually has a summation order to get wrong -- agrees to the last bit.
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyattn as A

graph = json.load(open("/tmp/attn_graph.json"))
weights = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn_weights.json")))
samples = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn_samples.json")))

# verify topological order ourselves rather than trusting it
seen = set()
for c in graph["cells"]:
    for d in c.get("inputs", []):
        if d not in seen:
            raise SystemExit(f"graph is not in topological order: {c['id']} reads {d}")
    seen.add(c["id"])

SOURCES = {"input", "weight"}
per, total, n = {}, 0.0, 0
for s in samples:
    env = {k: list(v) for k, v in weights.items()}
    for c in graph["cells"]:
        if c["kind"] == "input" and not c["params"].get("slot"):
            continue
    # bind inputs the way their evaluateForward does
    for c in graph["cells"]:
        if c["kind"] != "input":
            continue
        slot = c["params"].get("slot")
        if slot == "x": env[c["id"]] = list(s["x"])
        elif slot == "y": env[c["id"]] = list(s["y"])   # NOT [s["y"]]: that double-wraps, making L=1 and V=len(logits)
    for c in graph["cells"]:
        if c["kind"] in SOURCES and c["kind"] != "weight":
            pass
        if c["kind"] in SOURCES:
            if c["kind"] == "weight": env[c["id"]] = list(weights[c["id"]])
            continue
        args = [env[d] for d in c.get("inputs", [])]
        if c["kind"] == "stack":
            out = A.FNS["stack"](env, *args, c)
        else:
            out = A.FNS[c["kind"]](env, *args, c)
        for v in out:
            if v != v or v in (float("inf"), float("-inf")):
                raise SystemExit(f"cell {c['id']} produced a non-finite value")
        env[c["id"]] = out
        if c["id"] not in per:
            per[c["id"]] = A.array_digest(out)
    loss_id = next(c["id"] for c in graph["cells"] if c["kind"] == "xent")
    total += env[loss_id][0]; n += 1

node = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "node_attn_out.json")))
mine_loss = A.f64hex(total / n)

print("  ATTENTION CONFORMANCE — two runtimes, one convention, no shared source")
print(f"    loss f64hex  python : {mine_loss}")
print(f"    loss f64hex  node   : {node['loss_f64hex']}    {'IDENTICAL' if mine_loss==node['loss_f64hex'] else 'DIVERGES'}")
common = sorted(set(per) & set(node["per_cell"]))
same = sum(1 for k in common if per[k] == node["per_cell"][k])
print(f"    per-cell array digests : {same}/{len(common)} identical")
diff = [k for k in common if per[k] != node["per_cell"][k]]
if diff: print(f"    first divergences: {diff[:5]}")
ce = total / n
print()
print("    the replicated scalarSha bug, on THIS repo's own loss value:")
print(f"      sha256(bytes.fromhex(hex))        {A.scalar_sha_portable(ce)[:40]}")
print(f"      sha256(utf8(latin1decode(bytes))) {A.scalar_sha_as_written(ce)[:40]}")
print(f"      these differ: {A.scalar_sha_portable(ce) != A.scalar_sha_as_written(ce)}")
json.dump({"cells_common": len(common), "cells_identical": same,
           "loss_f64hex_py": mine_loss, "loss_f64hex_node": node["loss_f64hex"],
           "scalar_portable": A.scalar_sha_portable(ce),
           "scalar_as_written": A.scalar_sha_as_written(ce)},
          open("attn_conformance.json", "w"), indent=1)
