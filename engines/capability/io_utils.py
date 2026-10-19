"""Local, portable evidence I/O. Score access never implicitly loads labels."""
from __future__ import annotations
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
RADII = (5, 10, 15, 20, 25)

def require(value, message):
    if not value:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def check_output(path):
    path = Path(path).resolve()
    require(not any((p / 'MANIFEST.json').exists() for p in (path.parent, *path.parent.parents)),
            'output is inside a sealed package')

def write_json(path, value):
    path = Path(path)
    check_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    rows = list(rows)
    require(bool(rows), 'empty table: ' + str(path))
    path = Path(path)
    check_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)

def load_scores(root=ROOT):
    root = Path(root)
    lock = read_json(root / 'INPUTS_LOCK.json')
    for name, digest in lock['scores'].items():
        require(sha(root / name) == digest, 'changed score input: ' + name)
    scores = dict(np.load(root / 'inputs/scores.npz', allow_pickle=False))
    queries = read_json(root / 'inputs/queries.json')
    require(scores['qids'].tolist() == [q['qid'] for q in queries], 'query identity')
    return queries, scores

def load_labels(root=ROOT):
    root = Path(root)
    lock = read_json(root / 'INPUTS_LOCK.json')
    for name, digest in lock['labels'].items():
        require(sha(root / name) == digest, 'changed label input: ' + name)
    return np.load(root / 'inputs/geography.npz', allow_pickle=False)['geo']

def group(q):
    if q['route'] == 'NSAVP':
        return 'all_cross'
    a, b = q['condition'].split('_to_')
    return 'same' if a == b else 'morning_four' if 'M' in (a, b) else 'day_night'

def select(scores, qi, rep='SALAD', indices=None):
    n = int(scores['lengths'][qi])
    ix = np.arange(n) if indices is None else np.asarray(indices, dtype=int)
    require(len(ix) and np.all((ix >= 0) & (ix < n)), 'invalid candidate indices')
    suffix = 's' if rep == 'SALAD' else 'c'
    ids = scores['ids'][qi, ix]
    r, t = scores['dr_' + suffix][qi, ix], scores['dt_' + suffix][qi, ix]
    require(np.isfinite(r).all() and np.isfinite(t).all(), 'nonfinite real scores')
    require(len(np.unique(ids)) == len(ids), 'duplicate candidate IDs')
    return ids, r, t, ix

@lru_cache(maxsize=12)
def draw_data(route, root=ROOT):
    meta = read_json(Path(root) / 'inputs/draws.json')[route]
    weights = np.load(Path(root) / f'inputs/draws_{route}.npz', allow_pickle=False)['weights']
    return meta, weights

def paired_interval(rows, metric, denominator=None, root=ROOT):
    route = rows[0]['route']
    require(all(r['route'] == route for r in rows), 'mixed routes')
    meta, weights = draw_data(route, root)
    anchors = meta['anchors']
    ai = {a: i for i, a in enumerate(anchors)}
    num, den = np.zeros(len(anchors)), np.zeros(len(anchors))
    for row in rows:
        j = ai[str(row['anchor'])]
        num[j] += float(row[metric])
        den[j] += float(row[denominator]) if denominator else 1
    sn, sd = weights @ num, weights @ den
    valid = sd > 0
    bounds = np.quantile(sn[valid] / sd[valid] * 100, [.025, .975]) if valid.any() else [None, None]
    return dict(count=float(num.sum()), denominator=float(den.sum()),
                value_percent=float(num.sum() / den.sum() * 100) if den.sum() else None,
                ci95_low=None if bounds[0] is None else float(bounds[0]),
                ci95_high=None if bounds[1] is None else float(bounds[1]),
                valid_draws=int(valid.sum()), undefined_draws=int((~valid).sum()))

def summarize(rows, metrics, by=('representation', 'policy', 'radius_m')):
    from collections import defaultdict
    buckets = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in by)
        for name in dict.fromkeys(('all_records', row.get('group', group(row)), row['condition'])):
            buckets[(key, row['route'], name)].append(row)
        if group(row) != 'same' and group(row) != 'all_cross':
            buckets[(key, row['route'], 'all_cross')].append(row)
    result = []
    for (key, route, name), subset in buckets.items():
        base = dict(zip(by, key), route=route, group=name, queries=len(subset))
        for metric in metrics:
            result.append(dict(base, metric=metric, **paired_interval(subset, metric)))
        for metric in ('scalar_impossible', 'D_impossible'):
            if metric in metrics and metric in subset[0]:
                for den in ('original_failure', 'candidate_present_failure'):
                    marked = [dict(r, conditional=int(r[metric]) * int(r[den])) for r in subset]
                    result.append(dict(base, metric=metric + '_fraction_of_' + den,
                                       **paired_interval(marked, 'conditional', den)))
    return result
