"""Independent sorted-frontier oracle and Fraction scalar verification.

Does not import monotone.py, analyze.py or sensitivity.py. All real candidate
certificates are checked. The scalar Fraction reference is the separately
frozen implementation, never the production integer solver.
"""
from __future__ import annotations
from bisect import bisect_left, bisect_right
from collections import defaultdict
import json
import time
from io_utils import ROOT, PROJECT, require, load_scores, load_labels, read_csv, read_json, write_json, sha
import numpy as np
from scalar_reference import reference, self_test

def independent_monotone(r, t, p, keys):
    ns = sorted((float(r[i]), float(t[i]), int(keys[i])) for i in range(len(r)) if not p[i])
    xs = [n[0] for n in ns]; mins = [float('inf')]
    identical = {}
    for x, y, key in ns:
        mins.append(min(mins[-1], y))
        identical[x, y] = min(identical.get((x, y), key), key)
    result = dict(M_optimistic=False, M_stable=False, D_optimistic=False, D_stable=False)
    for i in np.flatnonzero(p):
        x, y, key = float(r[i]), float(t[i]), int(keys[i])
        left, right = bisect_left(xs, x), bisect_right(xs, x)
        strict = mins[left] < y
        weak = mins[left] <= y or mins[right] < y
        eq_earlier = identical.get((x, y), key) < key
        le_earlier = any(ny <= y and nk < key for _, ny, nk in ns[:right])
        result['M_optimistic'] |= not weak
        result['M_stable'] |= not (weak or eq_earlier)
        result['D_optimistic'] |= not strict
        result['D_stable'] |= not (strict or le_earlier)
    return result

def domains(scores, qi, policy, cert):
    n = int(scores['lengths'][qi]); ids = scores['ids'][qi, :n]
    if cert.get('setting'):
        ix = np.array(cert['candidate_indices'], dtype=int); keys = ix
    elif policy == 'C20':
        mapping = {int(b): j for j, b in enumerate(ids)}
        ix = np.array([mapping[int(b)] for b in scores['c20'][qi]]); keys = np.arange(20)
    elif policy.startswith('K'):
        ix = np.array(cert['candidate_indices'], dtype=int); keys = ix
    else:
        ix = np.arange(n); keys = ix
    return ix, keys

def sequential_certificates(filename, scores, geo, qi_by_id):
    start = time.monotonic(); counts = defaultdict(int)
    with filename.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            cert = json.loads(line); qi = qi_by_id[cert['qid']]
            ix, keys = domains(scores, qi, cert['policy'], cert)
            ids = scores['ids'][qi, ix]
            require(sha_array(ids) == cert['candidate_sha256'], 'certificate candidate binding')
            suffix = 's' if cert['representation'] == 'SALAD' else 'c'
            r = scores['dr_' + suffix][qi, ix]; t = scores['dt_' + suffix][qi, ix]
            p = geo[qi, ix] <= cert['radius_m']
            actual = independent_monotone(r, t, p, keys)
            require(all(actual[k] == cert['monotone'][k] for k in actual), 'independent monotone disagreement')
            require(len(cert['monotone']['certificates']) == int(p.sum()), 'positive certificate completeness')
            for item in cert['monotone']['certificates']:
                pi = item['positive_index']; require(bool(p[pi]), 'certificate not positive')
                for name in ('weak_dominator', 'strict_both_dominator', 'equal_earlier', 'weak_or_equal_earlier'):
                    ni = item[name]
                    if ni is None:
                        continue
                    require(not p[ni], 'witness not negative')
                    if name == 'weak_dominator':
                        good = r[ni] <= r[pi] and t[ni] <= t[pi] and (r[ni] < r[pi] or t[ni] < t[pi])
                    elif name == 'strict_both_dominator':
                        good = r[ni] < r[pi] and t[ni] < t[pi]
                    elif name == 'equal_earlier':
                        good = r[ni] == r[pi] and t[ni] == t[pi] and keys[ni] < keys[pi]
                    else:
                        good = r[ni] <= r[pi] and t[ni] <= t[pi] and keys[ni] < keys[pi]
                    require(good, 'false dominance witness')
                    counts['witnesses'] += 1
            if cert['scalar'] is not None:
                order = np.argsort(keys)
                stable = reference(r[order], t[order], p[order])
                optimistic = reference(r[order], t[order], p[order], optimistic=True)
                require(bool(stable['intervals']) == cert['scalar']['attainable'], 'Fraction stable mismatch')
                require(bool(optimistic['intervals']) == cert['scalar']['optimistic'], 'Fraction optimistic mismatch')
                counts['scalar_fraction'] += 1
            counts['conditions'] += 1
            if number % 10000 == 0:
                print('AUDIT', filename.name, number, 'seconds', round(time.monotonic() - start, 1), flush=True)
    return dict(counts, seconds=time.monotonic() - start)

def sha_array(a):
    import hashlib
    return hashlib.sha256(a.tobytes()).hexdigest()

def check_rows(rows):
    by_radius = defaultdict(list)
    for row in rows:
        for k in ('M', 'D'):
            require(int(row[k + '_stable']) <= int(row[k + '_optimistic']), 'tie ordering')
        require(int(row['equal_correct']) <= int(row['scalar_stable']) <= int(row['D_stable']) <= int(row['positive_available']), 'stable hierarchy')
        require(int(row['scalar_optimistic']) <= int(row['D_optimistic']), 'optimistic hierarchy')
        require(int(row['M_optimistic']) <= int(row['D_optimistic']), 'class inclusion')
        token = tuple(row[k] for k in ('qid', 'representation', 'policy')) + (row.get('setting', ''),)
        by_radius[token].append(row)
    for values in by_radius.values():
        values.sort(key=lambda r: int(r['radius_m']))
        for metric in ('positive_available', 'equal_correct', 'scalar_stable', 'scalar_optimistic', 'M_stable', 'M_optimistic', 'D_stable', 'D_optimistic'):
            seq = [int(r[metric]) for r in values]
            require(seq == sorted(seq), 'radius monotonicity: ' + metric)


def initialize_shard(root):
    global SHARD_SCORES, SHARD_GEO, SHARD_QI
    queries, SHARD_SCORES = load_scores(root)
    SHARD_GEO = load_labels(root)
    SHARD_QI = {q['qid']:i for i,q in enumerate(queries)}

def audit_shard(filename):
    return sequential_certificates(filename, SHARD_SCORES, SHARD_GEO, SHARD_QI)

def audit_certificates(filename, scores, geo, qi_by_id):
    from concurrent.futures import ProcessPoolExecutor
    import os
    workers = int(os.environ.get('PAPER_A_AUDIT_WORKERS','4'))
    directory = ROOT/'audit/shards'/filename.stem
    directory.mkdir(parents=True,exist_ok=False)
    paths = [directory/f'part-{i}.jsonl' for i in range(workers)]
    streams = [p.open('w',encoding='utf-8') for p in paths]
    try:
        with filename.open(encoding='utf-8') as source:
            for i,line in enumerate(source): streams[i%workers].write(line)
    finally:
        for stream in streams: stream.close()
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize_shard,initargs=(ROOT,)) as pool:
        results = list(pool.map(audit_shard,paths))
    counts = {key:sum(result.get(key,0) for result in results)
              for key in ('conditions','witnesses','scalar_fraction')}
    return dict(counts,workers=workers,seconds=time.monotonic()-start,
                scheduling='round-robin disjoint shards; unchanged per-certificate assertions')


def main():
    start = time.monotonic(); queries, scores = load_scores(); geo = load_labels()
    qi = {q['qid']: i for i, q in enumerate(queries)}
    require(self_test() == 11, 'Fraction reference self test')
    first = audit_certificates(ROOT / 'analysis/certificates.jsonl', scores, geo, qi)
    second = audit_certificates(ROOT / 'analysis/sensitivity_certificates.jsonl', scores, geo, qi)
    ability = read_csv(ROOT / 'analysis/query_results.csv'); sensitivity = read_csv(ROOT / 'analysis/sensitivity_queries.csv')
    check_rows(ability); check_rows(sensitivity)
    # Independently reconstruct every published summary numerator/denominator and spatial CI.
    summary_cells = 0
    for data, table, keys in [(ability, 'statistics.csv', ('representation', 'policy', 'radius_m')),
                             (sensitivity, 'sensitivity.csv', ('setting', 'representation', 'radius_m'))]:
        groups = defaultdict(list)
        for row in data:
            for name in set(('all_records', row['condition'], route_group(row))):
                groups[tuple(row[k] for k in keys) + (row['route'], name)].append(row)
            if route_group(row) not in ('same', 'all_cross'):
                groups[tuple(row[k] for k in keys) + (row['route'], 'all_cross')].append(row)
        draws = {}
        for route in ('SNU', 'Valley', 'NSAVP'):
            m = read_json(ROOT / 'inputs/draws.json')[route]
            draws[route] = (m['anchors'], np.load(ROOT / f'inputs/draws_{route}.npz')['weights'])
        for cell in read_csv(ROOT / 'analysis' / table):
            values = groups[tuple(cell[k] for k in keys) + (cell['route'], cell['group'])]
            metric = cell['metric']; denominator = None
            if '_fraction_of_' in metric:
                metric, denominator = metric.split('_fraction_of_')
            anchors, weights = draws[cell['route']]
            nums = np.array([sum(int(r[metric]) * (int(r[denominator]) if denominator else 1) for r in values if r['anchor'] == a) for a in anchors])
            dens = np.array([sum(int(r[denominator]) if denominator else 1 for r in values if r['anchor'] == a) for a in anchors])
            require(float(cell['count']) == nums.sum() and float(cell['denominator']) == dens.sum(), 'summary count mismatch')
            sn, sd = weights @ nums, weights @ dens; valid = sd > 0
            require(int(cell['undefined_draws']) == int((~valid).sum()), 'undefined denominator lost')
            if valid.any():
                ci = np.quantile(sn[valid] / sd[valid] * 100, [.025, .975])
                require(np.allclose(ci, [float(cell['ci95_low']), float(cell['ci95_high'])], rtol=0, atol=1e-11), 'paired interval mismatch')
            summary_cells += 1
    overlap = read_csv(ROOT / 'analysis/model_overlap_queries.csv')
    for r in overlap:
        require(sum(int(r[k]) for k in ('both_correct', 'a_only', 'b_only', 'both_wrong')) == 1, 'overlap partition')
        require(int(r['delta_correct']) == int(r['b_only']) - int(r['a_only']), 'overlap delta identity')
    protected = read_json(ROOT / 'PORTABLE_MEMBERS.json')
    for name, digest in protected.items():
        require(sha(ROOT / name) == digest, 'portable member changed: ' + name)
    receipt = dict(status='PASSED', ability=first, sensitivity=second, summary_cells=summary_cells,
        overlap_rows=len(overlap), protected_files=len(protected), seconds=time.monotonic() - start,
        implementation='sorted negative frontier + independent Fraction; raw CSV aggregation and paired draws')
    write_json(ROOT / 'audit/ABC_E_AUDIT.json', receipt)
    print(receipt, flush=True)

def route_group(q):
    if q['route'] == 'NSAVP':
        return 'all_cross'
    a, b = q['condition'].split('_to_')
    return 'same' if a == b else 'morning_four' if 'M' in (a, b) else 'day_night'

if __name__ == '__main__':
    main()
