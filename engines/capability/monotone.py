"""Exact finite-table attainability for the M and D score function classes.

No epsilon comparison is used: IEEE binary64 ordering is exact for the stored
finite values. Weak dominance excludes identical vectors by definition.
"""
from __future__ import annotations
import numpy as np

def solve_monotone(dr, dt, positive, tie_keys):
    r, t, p, keys = map(np.asarray, (dr, dt, positive, tie_keys))
    if (r.dtype != np.float64 or t.dtype != np.float64 or p.dtype != bool or
        any(a.ndim != 1 for a in (r, t, p, keys)) or len(r) == 0 or
        not len(r) == len(t) == len(p) == len(keys) or
        not np.isfinite(r).all() or not np.isfinite(t).all() or
        keys.dtype.kind not in 'iu' or len(np.unique(keys)) != len(keys)):
        raise ValueError('invalid score, positive mask or tie keys')
    ps, ns = np.flatnonzero(p), np.flatnonzero(~p)
    certificates = []
    for pi in ps:
        le = (r[ns] <= r[pi]) & (t[ns] <= t[pi])
        strict = (r[ns] < r[pi]) & (t[ns] < t[pi])
        weak = le & ((r[ns] < r[pi]) | (t[ns] < t[pi]))
        equal = (r[ns] == r[pi]) & (t[ns] == t[pi])
        earlier = keys[ns] < keys[pi]
        def witness(mask):
            found = ns[mask]
            return int(found[0]) if len(found) else None
        certificates.append(dict(positive_index=int(pi),
            weak_dominator=witness(weak), strict_both_dominator=witness(strict),
            equal_earlier=witness(equal & earlier), weak_or_equal_earlier=witness(le & earlier),
            M_optimistic=not bool(weak.any()), M_stable=not bool(weak.any() or (equal & earlier).any()),
            D_optimistic=not bool(strict.any()), D_stable=not bool(strict.any() or (le & earlier).any())))
    return {name: any(c[name] for c in certificates)
            for name in ('M_optimistic', 'M_stable', 'D_optimistic', 'D_stable')} | {
                'positive_count': int(len(ps)), 'certificates': certificates}
