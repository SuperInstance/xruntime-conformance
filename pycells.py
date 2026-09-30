"""
pycells.py — an INDEPENDENT implementation of the quilt cell convention, in Python.

This is not a port of quilt-nn. It is a second implementation written from the same
specification, and the point is to find out whether two runtimes that never shared source
produce the same bytes. The fleet's polyformalism claim has only ever been made about
STATIC ports of hand-written code. This is the first time it can be made about a live
computation.

SPEC, from SuperInstance/quilt-nn and SuperInstance/quilt-attention:
  cells      { id, kind, inputs[], params{} }   in an array whose order is topological
  kinds      input | weight | grad | tick  -- SOURCES, skipped in the forward pass
             sum      acc = 0; for each input: acc += env[input]
             product  acc = 1; for each input: acc *= env[input]
             act      actValue(params.fn, env[input[0]])
             loss     (env[pred] - env[target])**2
  digest     f64hex: IEEE-754 big-endian 8 bytes as lowercase hex, with -0 normalised
             to 0. loss_sha: sha256 over the 8 RAW bytes of that hex.

Everything here is portable arithmetic on purpose -- no engine RNG, no crypto entropy.
"""
from __future__ import annotations
import hashlib, math, struct

ACTS = ("tanh", "relu", "sigmoid", "gelu", "silu")


def f64hex(x: float) -> str:
    if not math.isfinite(x):
        raise ValueError(f"f64hex needs a finite number, got {x}")
    if x == 0.0:                 # also catches -0.0
        x = 0.0
    return struct.pack(">d", x).hex()


def loss_sha(loss: float) -> str:
    return hashlib.sha256(bytes.fromhex(f64hex(loss))).hexdigest()


def act_value(fn: str, x: float) -> float:
    if fn == "tanh":   return math.tanh(x)
    if fn == "relu":   return x if x > 0 else 0.0
    if fn == "sigmoid":return 1.0 / (1.0 + math.exp(-x))
    if fn == "silu":   return x / (1.0 + math.exp(-x))
    if fn == "gelu":   return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
    raise ValueError(f"unknown act {fn!r}")


SOURCES = {"input", "weight", "grad", "tick"}


def validate(graph: dict) -> None:
    seen = set()
    for c in graph["cells"]:
        if c["id"] in seen:
            raise ValueError(f"duplicate cell id {c['id']!r}")
        seen.add(c["id"])
        for d in c.get("inputs", []):
            if d not in seen:
                raise ValueError(f"cell {c['id']!r} reads {d!r} before it is defined "
                                 f"-- array order must be a topological order")
        if c["kind"] == "act" and c.get("params", {}).get("fn") not in ACTS:
            raise ValueError(f"act cell {c['id']!r}: fn must be one of {ACTS}")


def init_env(graph: dict, seed: int = 0) -> dict:
    """Weights get a deterministic LCG so two runtimes agree without a shared RNG."""
    s = seed & 0xFFFFFFFF
    env = {}
    for c in graph["cells"]:
        if c["kind"] == "weight":
            s = (1664525 * s + 1013904223) & 0xFFFFFFFF
            u = s / 4294967296.0
            scale = float(c.get("params", {}).get("scale", 1.0))
            env[c["id"]] = (u - 0.5) * 2.0 * scale
    return env


def forward(graph: dict, env: dict, x: list[float], y: float) -> dict:
    validate(graph)
    feats = [c for c in graph["cells"] if c["kind"] == "input" and not c.get("params", {}).get("target")]
    target = next((c for c in graph["cells"]
                   if c["kind"] == "input" and c.get("params", {}).get("target")), None)
    for k, c in enumerate(feats):
        v = x[c.get("params", {}).get("slot", k)]
        if not math.isfinite(v):
            raise ValueError(f"sample x[{c.get('params', {}).get('slot', k)}] must be finite")
        env[c["id"]] = float(v)
    if target is not None:
        env[target["id"]] = float(y)
    for c in graph["cells"]:
        k, ins = c["kind"], c.get("inputs", [])
        if k in SOURCES:
            continue
        if k == "sum":
            acc = 0.0
            for d in ins: acc += env[d]
        elif k == "product":
            acc = 1.0
            for d in ins: acc *= env[d]
        elif k == "act":
            acc = act_value(c["params"]["fn"], env[ins[0]])
        elif k == "loss":
            d = env[ins[0]] - env[ins[1]]
            acc = d * d
        else:
            raise ValueError(f'forward cannot evaluate kind "{k}" ({c["id"]})')
        if not math.isfinite(acc):
            raise ValueError(f"cell {c['id']} evaluated to a non-finite value")
        env[c["id"]] = acc
    return env


def digest_of(env: dict, cell_id: str) -> str:
    """The per-cell digest, in the documented form: the dtype is INSIDE the preimage."""
    return hashlib.sha256(f"f64|8|{f64hex(env[cell_id])}".encode()).hexdigest()
