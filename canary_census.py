#!/usr/bin/env python3
"""
canary_census.py — does every implementation of the fleet digest actually agree?

Four independent implementations of FNV-1a 64 now exist in the fleet, written in three
languages by (as far as we can tell) three different people, and the conformance of
*computation* has been verified for two of them against a second runtime. This checks the
hash itself against a reference and against the fleet canary.

The interesting row is the accent. "cafe Δ 日本語" without the accent hashes to something
else entirely, and it looks identical on screen. That trap has now been hit in the Futhark
canary, in the education site, and here — in a repo that had nothing to do with either.
"""
FNV_OFF = 14695981039346656037
FNV_PRIME = 1099511628211
MASK = (1 << 64) - 1
CANARY = 0x024a555471370b18d
FIXTURE = "café Δ 日本語"        # accented
UNACCENTED = "cafe Δ 日本語"      # NOT the canary


def fnv1a_64(s: str) -> int:
    h = FNV_OFF
    for b in s.encode("utf-8"):
        h = ((h ^ b) * FNV_PRIME) & MASK
    return h


VECTORS = [
    ("", 0xCBF29CE484222325, "the FNV-1a 64 offset basis is the empty-string hash"),
    ("a", 0xAF63DC4C8601EC8C, "published KAT, also asserted in micrograd-quilt's own suite"),
    ("foobar", 0x85944171F73967E8, "classic FNV-1a 64 test vector"),
    (FIXTURE, CANARY, "the fleet canary"),
]

IMPLS = [
    ("micrograd-quilt", "Python", "quilt/tape.py fnv1a_64", "hash chain over witness rows"),
    ("cellgraph",       "Python", "this file",                "per-cell tensor digest uses BLAKE2b; the FNV is the port canary"),
    ("QuiltCanary.jl",  "Julia",  "src/QuiltCanary.jl",       "13/13 tests"),
    ("canary-3lang",    "Futhark","canary.fut",               "VERIFIED by execution, 0x24a555471370b18d"),
    ("audit-trail",     "Rust",   "src/lib.rs fnv1a64",      "14/14 tests, hash chain over receipts"),
]

if __name__ == "__main__":
    print("  FNV-1a 64 — reference vs the fleet canary, and the accent trap\n")
    allok = True
    for s, want, why in VECTORS:
        got = fnv1a_64(s)
        ok = got == want
        allok &= ok
        print("    %-8s 0x%016X  %s" % (repr(s)[1:-1][:8], got, "ok" if ok else "MISMATCH"))
        print("             expected 0x%016X   %s" % (want, why))
    print()
    u = fnv1a_64(UNACCENTED)
    print(f"    UNACCENTED fixture 0x{u:016X}")
    print(f"    is it the canary?   {u == CANARY}  <- the trap")
    print()
    print("  implementations in the fleet:")
    for name, lang, where, use in IMPLS:
        print(f"    {name:20} {lang:8} {where:26} {use}")
    print()
    print(f"  reference vectors: {'ALL PASS' if allok else 'FAILURES PRESENT'}")
