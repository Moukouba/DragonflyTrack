"""Identity memory module for DragonflyTrack-X."""

import numpy as np


class IdentityMemory:
    def __init__(
        self,
        dim,
        expand=4,
        sparsity=0.1,
        eta0=0.2,
        bank=8,
        seed=0,
        min_quality=0.12,
    ):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 1 / np.sqrt(dim), (expand * dim, dim))
        self.k = max(1, int(sparsity * expand * dim))
        self.eta0, self.bank_size, self.min_q = eta0, bank, min_quality
        self.mu, self.bank = {}, {}

    def _expand(self, f):
        f = np.asarray(f, float).ravel()
        n = np.linalg.norm(f)
        if n < 1e-12:
            return np.zeros(self.W.shape[0])
        z = self.W @ (f / n)
        idx = np.argpartition(-np.abs(z), self.k)[: self.k]
        out = np.zeros_like(z)
        out[idx] = z[idx]
        nn = np.linalg.norm(out)
        return out / nn if nn > 1e-12 else out

    def has(self, tid):
        return tid in self.mu

    def update(self, tid, feat, quality):
        q = float(np.clip(quality, 0.0, 1.0))
        if q < self.min_q:
            return
        z = self._expand(feat)
        if tid not in self.mu:
            self.mu[tid] = z.copy()
            self.bank[tid] = [z]
            return
        eta = self.eta0 * q
        mu = (1 - eta) * self.mu[tid] + eta * z
        self.mu[tid] = mu / max(np.linalg.norm(mu), 1e-12)
        self.bank[tid].append(z)
        b = self.bank[tid]
        if len(b) > self.bank_size:
            B = np.stack(b)
            S = B @ B.T
            np.fill_diagonal(S, -np.inf)
            i, j = np.unravel_index(np.argmax(S), S.shape)
            del b[max(i, j)]

    def similarity(self, tid, feat, z=None):
        if tid not in self.mu:
            return 0.0
        if z is None:
            z = self._expand(feat)
        return float(
            max(z @ self.mu[tid], (np.stack(self.bank[tid]) @ z).max())
        )

    def expand_many(self, F):
        return [self._expand(f) for f in F]

    def repel(self, a, b, strength=0.03):
        if a not in self.mu or b not in self.mu:
            return
        pa, pb = self.mu[a], self.mu[b]
        d = pa - pb
        n = np.linalg.norm(d)
        if n < 1e-9:
            return
        pa2, pb2 = pa + strength * d / n, pb - strength * d / n
        self.mu[a] = pa2 / max(np.linalg.norm(pa2), 1e-12)
        self.mu[b] = pb2 / max(np.linalg.norm(pb2), 1e-12)

    def forget(self, tid):
        self.mu.pop(tid, None)
        self.bank.pop(tid, None)
