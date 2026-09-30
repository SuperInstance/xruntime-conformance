"""
pyattn.py — an INDEPENDENT Python implementation of the attention cell convention.

Same intent as pycells.py, for the array-valued cells in SuperInstance/quilt-attention.

THE DETAIL THAT DECIDES WHETHER THIS CAN WORK AT ALL
-----------------------------------------------------
`matmul` accumulates with a SPECIFIC naive loop order:

    for i:            # rows
        for k:        # inner dim
            if X[i*p+k] == 0: continue      # exact-zero skip
            for j: out[i*c+j] += X[i*p+k] * W[k*c+j]

That is an outer-product accumulation, NOT `X @ W` with a blocked/vectorised sum. A
numpy matmul sums in a different order and produces different bits. So this is written as
plain Python loops in the SAME order. Slow on purpose: the question is whether two runtimes
agree to the last bit, and numpy would answer a different question.

`attn` is softmax over rows of (Q·Kᵀ / sqrt(dim)) with no causal mask.
"""
from __future__ import annotations
import hashlib, math, struct

DTYPE_TAG = "f64"


def f64hex(x: float) -> str:
    if not math.isfinite(x):
        raise ValueError(f"not finite: {x}")
    if x == 0.0:
        x = 0.0                      # also normalises -0.0
    return struct.pack(">d", x).hex()


def array_digest(a) -> str:
    """dtype tag + length + bytes, dtype INSIDE the preimage. This is the form that
    survives a second runtime; see CONFORMANCE.md for the one that does not."""
    body = "".join(f64hex(v) for v in a)
    return hashlib.sha256(f"{DTYPE_TAG}|{len(a)}|{body}".encode()).hexdigest()


def scalar_sha_portable(x: float) -> str:
    """The portable reading of 'sha256 over the canonical bytes of one scalar'."""
    return hashlib.sha256(bytes.fromhex(f64hex(x))).hexdigest()


def scalar_sha_as_written(x: float) -> str:
    """What quilt-nn's lossShaOf and quilt-attention's scalarSha actually do:
    bytes -> latin1 string -> UTF-8 encode. Kept so the difference is demonstrable."""
    return hashlib.sha256(bytes.fromhex(f64hex(x)).decode("latin1").encode("utf-8")).hexdigest()


def softmax_rows(s, rows, cols):
    out = [0.0] * (rows * cols)
    for i in range(rows):
        off = i * cols
        row = s[off:off + cols]
        m = max(row)
        ex = [math.exp(v - m) for v in row]
        z = 0.0
        for e in ex: z += e
        for j in range(cols):
            out[off + j] = ex[j] / z
    return out


# ── the cells ────────────────────────────────────────────────────────────────────
def c_input(env, cell):
    v = env.get(cell["id"])
    if v is None: raise ValueError(f"input cell {cell['id']} has no bound value")
    return v

def c_weight(env, cell):
    w = env.get(cell["id"])
    if w is None: raise ValueError(f"weight cell {cell['id']} not initialized")
    return w

def c_embed(env, tokens, table, cell):
    pos = cell["params"]["pos"]; width = cell["params"]["width"]
    if not (0 <= pos < len(tokens)): raise ValueError(f"embed {cell['id']}: pos out of range")
    tok = tokens[pos]
    if not isinstance(tok, int) or tok < 0: raise ValueError(f"embed {cell['id']}: bad token {tok}")
    if (len(table) % width) or tok * width >= len(table):
        raise ValueError(f"embed {cell['id']}: token {tok} outside table")
    return table[tok * width:(tok + 1) * width]

def c_posemb(env, posTable, cell):
    pos = cell["params"]["pos"]; width = cell["params"]["width"]
    seqLen = len(posTable) // width
    if not (0 <= pos < seqLen): raise ValueError(f"posemb {cell['id']}: pos {pos} outside table")
    return posTable[pos * width:(pos + 1) * width]

def c_add(env, a, b, cell):
    if len(a) != len(b): raise ValueError(f"add {cell['id']}: length mismatch")
    return [a[i] + b[i] for i in range(len(a))]

def c_stack(env, *rest):
    cell = rest[-1]; rows = list(rest[:-1])
    if not rows: raise ValueError(f"stack {cell['id']}: no inputs")
    w = len(rows[0])
    out = []
    for r in rows:
        if len(r) != w: raise ValueError(f"stack {cell['id']}: rows of unequal length")
        out.extend(r)
    return out

def c_matmul(env, X, W, cell):
    rows = cell["params"]["rows"]
    if rows <= 0 or len(X) % rows: raise ValueError(f"matmul {cell['id']}: bad X length")
    p = len(X) // rows
    if len(W) % p: raise ValueError(f"matmul {cell['id']}: bad W length")
    c = len(W) // p
    out = [0.0] * (rows * c)
    for i in range(rows):                     # same order, same zero-skip
        for k in range(p):
            xv = X[i * p + k]
            if xv == 0: continue
            woff = k * c
            for j in range(c):
                out[i * c + j] += xv * W[woff + j]
    return out

def c_attn(env, Q, K, cell):
    L = cell["params"]["len"]; dim = cell["params"]["dim"]
    if len(Q) != L * dim or len(K) != L * dim:
        raise ValueError(f"attn {cell['id']}: Q/K length wrong")
    S = [0.0] * (L * L)
    scale = 1.0 / math.sqrt(dim)
    for i in range(L):
        for j in range(L):
            s = 0.0
            for k in range(dim):
                s += Q[i * dim + k] * K[j * dim + k]
            S[i * L + j] = s * scale
    return softmax_rows(S, L, L)

def c_xent(env, logits, target, cell):
    L = len(target)
    if len(logits) % L: raise ValueError(f"xent {cell['id']}: bad logits length")
    V = len(logits) // L
    ce = 0.0
    for i in range(L):
        t = target[i]
        if not isinstance(t, int) or not (0 <= t < V):
            raise ValueError(f"xent {cell['id']}: target {t} out of vocab {V}")
        off = i * V
        m = max(logits[off:off + V])
        z = 0.0
        for j in range(V): z += math.exp(logits[off + j] - m)
        ce += -(logits[off + t] - m - math.log(z))
    return [ce / L]

FNS = {"input": c_input, "weight": c_weight, "embed": c_embed, "posemb": c_posemb,
       "add": c_add, "stack": c_stack, "matmul": c_matmul, "attn": c_attn, "xent": c_xent}
