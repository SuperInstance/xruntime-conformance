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

---

# Part 2 — attention, and the same bug in a second repo

`SuperInstance/quilt-attention` is the array-valued sibling: cells produce `Float64Array`,
and the kinds are `posemb`, `stack`, `matmul`, `attn`, `xent`. Its README states it inherits
`cellgraph`'s convention and documents a deliberate deviation — no blake2b in Node, so sha256
with the dtype **inside** the preimage — and names that as "Rule 2."

**That deviation is the right call, and it is why the 11 cells agree.**

## The result

`pyattn.py` is an independent Python implementation written from the spec. `node_attn_runner.mjs`
calls their `evaluateForward` unmodified. Weights pinned, not seeded. One head, `len=3`,
`dim=4`, one positional table — so the arithmetic actually includes a summation order:

```
loss f64hex  python : 3ff63c3088de7d5f
loss f64hex  node   : 3ff63c3088de7d5f    IDENTICAL, bit for bit
per-cell array digests : 11/11 identical
```

Through `posemb` ×3, `stack`, `matmul` ×4, `attn` (Q·Kᵀ/√d + row softmax), and `xent`.
**Two runtimes, no shared source, bit-identical through a real attention forward pass.**

## The method detail that makes it possible

`matmul` accumulates in a specific naive order:

```js
for (i) for (k) { if (X[i*p+k] === 0) continue; for (j) out[i*c+j] += xv * W[k*c+j]; }
```

That is an outer-product accumulation, not a blocked or vectorised sum. **A numpy matmul
on the Python side would have summed in a different order and produced different bits** —
and the conclusion would have been "the runtimes disagree," which would have been a statement
about BLAS threading rather than about the convention. `pyattn.py` is plain Python loops in
the same order. Slow on purpose.

Worth writing down: **if you try this with a fast matrix library and the runtimes disagree,
the library is the most likely explanation.**

## The bug is in both repos, character for character

```js
// quilt-attention  src/attncells.mjs:78  (scalarSha)
// quilt-nn         src/cellgraph.mjs     (lossShaOf)
return sha256hex(Buffer.from(f64hex(x), 'hex').toString('latin1'));
```

`toString('latin1')` produces a **string**, and `.update(str)` encodes it as **UTF-8**, so
every byte ≥ 0x80 becomes two bytes. On this repo's own loss value:

```
sha256(bytes.fromhex(hex))          778058226d15f9a85ea68b2ffb6178225f2553c7
sha256(utf8(latin1decode(bytes)))   e41abcd419d16d369d0e6d393ed1a342b4e15497
```

Two implementations that both state they share no source made the identical mistake, which
suggests it is simply the natural way to read *"turn these hex bytes into something hashable."*
`arrayDigest` in the same file is correct, and is the fix in miniature:

```js
sha256hex(`${DTYPE_TAG}|8|${f64hex(x)}`)   // legible string preimage, dtype inside
```

Filed as [`quilt-attention` issue #1](https://github.com/SuperInstance/quilt-attention/issues/1),
alongside [`quilt-nn` issue #1](https://github.com/SuperInstance/quilt-nn/issues/1).

## The scoreboard, after three runtimes

| artefact | verdict |
|---|---|
| cell computation, scalars (`quilt-nn`) | **23/23 digests, loss bit-identical** |
| cell computation, arrays (`quilt-attention`) | **11/11 digests, loss bit-identical** |
| `arrayDigest` (dtype inside the preimage) | **portable** |
| `scalarSha` / `lossShaOf` (latin1 → UTF-8) | **not portable — replicated in two repos** |
| `evaluateForward` on a cold env | throws; `initEnv` referenced but not exported |

**The convention is real. Two of its digest definitions are not, and neither had a
cross-runtime test until today.** That is the whole argument for building a second
implementation: both bugs are invisible to a test written by the same person who wrote the
string, and one of them was replicated verbatim into a repo that documents itself as
sharing no source with the other.

---

# Part 3 — the fourth implementation, and a fleet-wide canary census

`SuperInstance/micrograd-quilt` is a different shape from the three above: it is an autograd
engine with a **witness tape** (`quilt/tape.py`) — an append-only FNV-1a 64 hash chain over
opcode rows (BIND, LINK, EFFECT, VIEW, TICK, FORGET). So it exercises a different part of
the doctrine: the chain, not the cell digest.

It also ships its own known-answer vectors, which makes it checkable in one line.

## It agrees, and it honours its own KAT

```
'café Δ 日本語'  micrograd-quilt  0x24A555471370B18D    == canary
'a'                              0xAF63DC4C8601EC8C    == its published KAT
''                               0xCBF29CE484222325    == the FNV offset basis
'foobar'                         0x85944171F73967E8    == classic test vector
'abc'                            0xE71FA2190541574B
'quilt'                          0x4B7FB2143373CE1E
```

**6/6 identical to an independent implementation**, the published KAT is honoured, and the
fleet canary holds.

## The accent trap, third sighting

```
'café Δ 日本語'  ->  0x24A555471370B18D    the canary
'cafe Δ 日本語'  ->  0xFEE91CF40962B966    not the canary
```

Same shape on screen, different test. It has now been hit in the Futhark canary, in the
education site, and here — in a repository that had nothing to do with either of them. That
is three sightings across three languages and three unrelated pieces of work, which makes it
a property of the fixture rather than a mistake anyone made once.

**It belongs in the fixture's own name.** A constant called `CANARY` and a fixture called
`"cafe Δ 日本語"` will eventually be paired wrongly, and nothing in the code will complain.
Naming it `FIXTURE_ACCENTED_CANARY` costs nothing and removes the ambiguity at the point
of use.

## Fleet census

| implementation | language | what it hashes | agrees? |
|---|---|---|---|
| `micrograd-quilt` | Python | witness tape chain | **yes**, KAT honoured, canary holds |
| `cellgraph` | Python | port canary (BLAKE2b for tensors) | **yes** |
| `QuiltCanary.jl` | Julia | port canary | **yes**, 13/13 |
| `canary-3lang` (Futhark) | Futhark | port canary | **yes**, verified by execution |
| `audit-trail` | Rust | receipt chain | **yes**, 14/14 |
| `quilt-nn` | JavaScript | `lossShaOf` | **NO** — latin1→UTF-8 |
| `quilt-attention` | JavaScript | `scalarSha` | **NO** — identical bug |

**Five of seven agree. The two that don't are the two that share a bug rather than a
convention.** Everything the fleet wrote independently agrees; the two that diverge are the
two an agent copied between.

That is a clean, slightly uncomfortable result: **the convention is robust and the copy is
not.** Two implementations that document themselves as sharing no source made the same
mistake, which means the mistake travels by being read, not by being invented.

---

# Part 4 — a fork in the doctrine, found by reading rather than testing

`micrograd-quilt` does **not** use the cell-graph convention. Zero files in it contain a
`"kind"` field. It records the *same six opcodes* in a completely different data structure:

```python
def bind(self,  v): return self._emit({"t":"BIND",   "id":v.id, "data":v.data})
def link(self,  v): return self._emit({"t":"LINK",   "id":v.id, "op":v._op, "p":[p.id for p in v._parents]})
def effect(self, node, parent, g): return self._emit({"t":"EFFECT","node":node.id,"parent":parent.id,"g":g})
def tick(self, step, v): return self._emit({"t":"TICK","step":step,"node":v.id})
def view(self, node, **kw): return self._emit({"t":"VIEW","node":node.id, **kw})
def forget(self, before): return self._emit({"t":"FORGET","before":before})
```

**A tape, not a graph.** The graph is *reconstructed* by accumulation from the ordered log.

## The two forms buy different things, and the difference is not cosmetic

| | tape (`micrograd-quilt`) | cell graph (`cellgraph`, `quilt-nn`, `quilt-attention`) |
|---|---|---|
| order | **fixed and recorded** — the log *is* the order | **derived** — array order is merely a topological one |
| re-evaluate a subgraph | impossible; the tape is a prefix history | **trivial** — evaluate any cell alone |
| forward replay | exact, by construction | exact, if you re-derive the order |
| random access to a past value | only by replaying to that point | direct |

**The consequence is concrete and already demonstrated.** `find_fault` — the routine that
evaluates a suspect cell and diffs witness digests downstream to localise a fault — **only
works on the cell-graph form.** It is a random-access operation on the graph. A tape cannot
support it, because the tape's guarantee is exactly that the order is fixed.

So the fleet has, without either implementation being wrong:

- **a graph form**, which supports localisation, fault attribution, and evaluation of one
  cell in isolation, at the cost of having to specify a topological order and be honest that
  it is one;
- **a tape form**, which supports exact replay and is simpler to append to, at the cost of
  giving up everything that requires looking sideways.

## What this should settle, not start

This is a fork in the doctrine and it should be a **decision, not a merge**. The proposal:

> **A cell graph is the canonical artefact. A tape is a rendering of one.**

That preserves both: `micrograd-quilt`'s tape is exactly what you would generate *from* a
cell graph for replay, and its `_canon` serialisation plus FNV chain is a perfectly good
replay digest. Nothing is lost, and the fault-localisation capability is not duplicated
across two incompatible ways.

**The test for whether a thing is a cell or a tape, and it is sharp:** *can you evaluate one
cell in isolation, without replaying everything before it?* If yes, it is a graph. If it can
only be run front to back, it is a tape. Both are legitimate; they are just not the same
artefact, and calling them the same thing is how you end up unable to localise a fault in
a system that has one.

---

# Part 5 — a correction to Part 1, and the third occurrence of the bug

**I was wrong about the origin of the `latin1` bug.** I wrote:

> "Two implementations that both state they share no source made the identical mistake,
> which suggests it is the natural way to read 'turn these hex bytes into something
> hashable.'"

A fleet-wide sweep found a **third** occurrence:

| repo / line | status |
|---|---|
| `quilt-nn`/`src/cellgraph.mjs`:**65** | origin — `lossShaOf`, called from `train.mjs:61,150` |
| `quilt-attention`/`src/attncells.mjs`:**78** | origin — `scalarSha`, called from `train.mjs:136,137,208,213` |
| `quilt-ml-recipes`/`recipes/r2-cellgraph-mlp-training/index.mjs`:**45** | **declared copy** — its own header, lines 9–10, says it was copied from `quilt-nn @ 1ae7977` |

So it is **two points of origin, not two convergent mistakes**, and the third is documented
as a copy. My "two independent implementations made the same mistake" was a story I liked
more than the evidence supported. The honest version is worse and more useful: **the bug
travels by being read.** One mistake, and something that documents itself as having read it.

**The blast radius is not cosmetic either.** Those digests chain into receipts that are
re-verified on replay, so the non-portable step sits in the **integrity path of the training
loop**. Fixing the source at `quilt-nn/src/cellgraph.mjs:65` and the acknowledged
derivative fixes all three.

`quilt-rl`/`src/qlearn.mjs`:36,43–51 has `f64hex` + `sha256` and joins as UTF-8 — the
**correct** pattern, independently written, in the same fleet. `quilt-ewitness`/`src/train.mjs`:12
explicitly disclaims the pattern. So the fleet knows the right form and reaches for the
wrong one anyway about a third of the time.

## The sweep also found a live defect, in the package that owns the canon

`substrate-foundation` — inherited by every `substrate-*` repo — destructured **eleven
per-opcode names** out of `@superinstance/opcode-canon`, which exports **arrays**:

```
ALL_OPCODES  = ["BIND","LINK","EFFECT","VIEW","TICK",
                "ATTEST","DELEGATE","CONTEST","MERGE","REVOKE","WITHDRAW"]   correct
BIND         = undefined      ATTEST  = undefined      MERGE  = undefined
LINK         = undefined      DELEGATE= undefined      REVOKE = undefined
EFFECT       = undefined      CONTEST = undefined      WITHDRAW=undefined
VIEW         = undefined
TICK         = undefined      MERGER  = undefined  (never an opcode; canonical is MERGE)
```

**All eleven were `undefined` at runtime** and nothing threw, because they were only
re-exported rather than used in logic there. A consumer doing
`const { BIND } = require('substrate-foundation')` got `undefined`, silently.

Fixed — the eleven are now derived from `ALL_OPCODES` and cannot drift from it — with
`test/exports.test.js` asserting that every name is *defined*, that the set is exactly the
canon set, that each agrees with the canon package's signatures, and that `MERGER` is absent.
**The test that would have caught this is "does it load", and that is not the question.**
Verified from a fresh clone: 0 undefined, 5/5 pass.

## The lesson the fleet has not drawn yet

The canary is applied to the **fixture string** and never to the **opcode names**. The
conservation law is checked where the numbers are; **the alphabet is unchecked.** That is
precisely how `MERGER` survived in the package that defines the canon — and it is why the
fix above had to be "derive it" rather than "correct the spelling," since a spelled
correction would leave the other ten broken and the bug looking fixed.

**Proposal: a canary over the alphabet.** FNV-1a 64 over the sorted `ALL_OPCODES` joined by
a fixed separator, asserted in CI. It is one line, it is cheap, and it would have failed the
day this shipped. A digest over the *words* the substrate is made of, not only over the
bytes it hashes.
