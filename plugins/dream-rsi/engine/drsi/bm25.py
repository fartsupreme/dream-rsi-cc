"""Okapi BM25 over short attempt fingerprints (stdlib only).

This is the cheap prefilter in front of the novelty judge: it picks the few
prior attempts worth showing the judge, so the judge never needs the whole
history in its context.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_STOP = frozenset("""
a an and are as at be been but by can could did do does for from had has have if in into is it its
of on or our so such than that the their then there these they this those to was were which while
with would we you your not no yes via per over under about after before between both each more most
""".split())

_TOKEN = re.compile(r"#\d+|\d+\^\d+|[^\W_]+(?:-[^\W_]+)*")  # any script, not just ASCII


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall((text or "").lower()):
        if tok in _STOP:
            continue
        out.append(tok)
        if "-" in tok and not tok.startswith("#"):
            out.extend(p for p in tok.split("-") if p and p not in _STOP)
    return out


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = list(docs)
        self.tfs = [Counter(tokenize(docs[i])) for i in self.ids]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def top_k(self, query: str, k: int) -> list[tuple[str, float]]:
        q = [t for t in tokenize(query) if t in self.idf]
        if not q or not self.ids:
            return []
        scored = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avgdl or 1))
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                scored.append((self.ids[i], s))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
