# Cross-runtime conformance: the cell convention, in two runtimes

The fleet's polyformalism claim has only ever been about **static ports of hand-written
code**. `quilt-nn`, `quilt-attention` and `cellgraph` are three independent
implementations of the same cell convention — Python, JavaScript, JavaScript — which
makes it possible to make the claim about something else: **a live computation.**

## What was run

The same graph, `SuperInstance/quilt-nn`'s `nets/xor.json` (61 cells), evaluated twice:

- **Node**, using *their* `evaluateForward` from `src/cellgraph.mjs`, unmodified.
- **Python**, using `pycells.py` here, written from the specification and sharing no source
  with theirs.

Weights were **pinned to explicit values** rather than seeded, so this measures *evaluation*
agreement and not RNG agreement. 23 forward-reachable cells, 4 XOR samples.

## The result, which is two results

```
loss f64hex,  python : 3fd113560f9094f0
loss f64hex,  node   : 3fd113560f9094f0        IDENTICAL — bit-for-bit
per-cell digests     : 23/23 identical
loss_sha,    python  : a1ecb2f6...
loss_sha,    node    : ce231e5e...             DIVERGES
```

**The computation is portable.** Every cell digest matches, and the mean loss is the same
float64 to the last bit, across two runtimes that have never shared a line of code. That is
a stronger polyformalism result than a hand-ported canary, because it covers arithmetic,
not just string encoding.

**The loss digest is not**, and the divergence is exactly located:

```js
// quilt-nn
sha256hex(Buffer.from(f64hex(loss), 'hex').toString('latin1'))
//   ^ hashes the UTF-8 encoding of a STRING whose code points are the 8 bytes

// the obvious reading of the same docstring
sha256(bytes.fromhex(f64hex(loss)))
//   ^ hashes the bytes
```

`toString('latin1')` produces a string; `.update(str)` then encodes it as **UTF-8**, and
every byte ≥ 0x80 becomes two bytes. The two agree only when all 8 bytes are ASCII, which
for a float64 never happens. **So `lossShaOf` does not hash the canonical bytes its own
docstring says it does** — it is a portable-looking digest that is not portable.

This is exactly the dtype-in-preimage class from the cellgraph digest discipline, one level
up: *the preimage is not what the documentation says it is.* `quilt-attention` already fixed
this for its own array digest by putting the dtype **inside** the preimage; `lossShaOf` has
the same bug and has not been touched.

**Recommendation:** hash the bytes. If a string preimage is wanted for legibility, write
it as `f64|8|<hex>` and hash *that string* — the per-cell digest already does exactly this
and it is why the 23 cells agree.

## A second finding: the forward pass cannot be called cold

```
6-cell graph, no grad/tick cells   -> OK
same graph + one grad cell         -> THROWS  "cell g_w evaluated to a non-finite value"
```

`evaluateForward` `break`s on `grad` and `tick` (they are set by backward and by the
training loop) and then runs a finiteness check over them, which is `undefined`. Also
`initEnv` is referenced in an error message but **is not exported** by `cellgraph.mjs`.

**Their own tests pass 12/12**, so this is latent, not active — the training loop warms the
env first. But it means the forward pass has an **undocumented precondition**, and any
cold-start caller — which is exactly what a conformance harness is — has to discover it by
crashing. Adding `if (c.kind in SOURCES) continue;` before the finiteness check, or
exporting a real `initEnv`, would fix it.

## What this means for the fleet

Three independent implementations of one convention, and **they agree on computation and
disagree on a digest.** That is the right order of news. The convention is real; two of its
digest definitions are not portable, and neither had a cross-runtime test until now.

The generalisable lesson is the one the fleet already wrote down: **Rule 2 of the cellgraph
digest discipline is that the preimage must be explicit.** This is Rule 3 — *the preimage
must be the thing the docstring claims it is*, checked by a second implementation rather
than by reading.
