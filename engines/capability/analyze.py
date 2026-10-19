"""A/B: candidate-bound exact scalar and monotone ability; C: model overlap."""
from __future__ import annotations
import itertools
import time
from collections import defaultdict
from io_utils import *
from monotone import solve_monotone
from scalar_exact import solve_query

METRICS = ('positive_available', 'equal_correct', 'scalar_stable', 'scalar_optimistic',
           'M_stable', 'M_optimistic', 'D_stable', 'D_optimistic',
           'scalar_impossible', 'D_impossible', 'scalar_headroom', 'D_extra_headroom')

def record(q, qi, rep, policy, ids, r, t, ix, geo, keys, radius, old, fixed303):
    positive = geo[qi, ix] <= radius
    mono = solve_monotone(r, t, positive, keys)
    equal_local = int(np.lexsort((keys, (r + t) / 2))[0])
    if old is None:
        scalar = solve_query(r, t, positive, keys)
        stable, optimistic = scalar['attainable'], scalar['optimistic']
    else:
        scalar = None
        stable, optimistic = bool(int(old['continuous_oracle'])), bool(int(old['optimistic_oracle']))
        require(int(positive[equal_local]) == int(old['equal_correct']), 'legacy equal replay')
        require(int(positive.any() and not mono['M_optimistic']) == int(old['every_positive_weakly_negative_dominated']), 'weak replay')
        require(int(positive.any() and not mono['D_optimistic']) == int(old['every_positive_strictly_both_negative_dominated']), 'strict-both replay')
    require(not stable or mono['D_stable'], 'stable scalar is not included in D')
    require(not optimistic or mono['D_optimistic'], 'optimistic scalar is not included in D')
    equal = bool(positive[equal_local])
    require(not equal or stable, 'equal excluded by scalar')
    available = bool(positive.any())
    classification = ('NO_POSITIVE' if not available else 'EQUAL_CORRECT' if equal else
                      'SCALAR_OPPORTUNITY' if optimistic else 'NONLINEAR_D_OPPORTUNITY' if mono['D_optimistic']
                      else 'STRICT_BOTH_OBSTRUCTION')
    row = {k: q[k] for k in ('qid', 'route', 'condition', 'anchor', 'block')}
    row.update(representation=rep, policy=policy, radius_m=radius, gallery_size=len(ids),
        positive_count=int(positive.sum()), positive_available=int(available), equal_correct=int(equal),
        scalar_stable=int(stable), scalar_optimistic=int(optimistic),
        **{k: int(mono[k]) for k in ('M_stable', 'M_optimistic', 'D_stable', 'D_optimistic')},
        scalar_impossible=int(available and not optimistic), D_impossible=int(available and not mono['D_optimistic']),
        scalar_headroom=int(optimistic) - int(equal), D_extra_headroom=int(mono['D_optimistic']) - int(optimistic),
        original_failure=int(not equal), candidate_present_failure=int(available and not equal),
        fixed303=int(q['qid'] in fixed303), equal_top1=int(ids[equal_local]),
        equal_distance_m=float(geo[qi, ix[equal_local]]), classification=classification)
    cert = dict(qid=q['qid'], representation=rep, policy=policy, radius_m=radius,
                candidate_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                candidate_indices=ix.tolist() if policy.startswith('K') else None,
                monotone=mono, scalar=scalar)
    return row, cert

def ability():
    start = time.monotonic()
    queries, scores = load_scores()
    geo = load_labels()
    prior = read_csv(ROOT / 'inputs/legacy_results.csv')
    old = {(r['qid'], r['representation'], r['policy'], int(r['radius_m'])): r for r in prior}
    fixed303 = {r['qid'] for r in prior if r['representation'] == 'SALAD' and r['policy'] == 'full'
                and r['route'] == 'SNU' and int(r['radius_m']) == 10 and group(r) == 'morning_four'
                and int(r['eligible_C_error_C20'])}
    require(len(fixed303) == 303, 'fixed subgroup identity')
    out = ROOT / 'analysis'
    out.mkdir(exist_ok=True)
    rows, prefix_checks = [], 0
    with (out / 'certificates.jsonl').open('w', encoding='utf-8') as f:
        for qi, q in enumerate(queries):
            n = int(scores['lengths'][qi]); full_ix = np.arange(n)
            lookup = {int(v): i for i, v in enumerate(scores['ids'][qi, :n])}
            c20_ix = np.array([lookup[int(v)] for v in scores['c20'][qi]])
            for rep in ('SALAD', 'Clique'):
                ids, r, t, _ = select(scores, qi, rep)
                ranking = np.argsort((r + t) / 2, kind='stable')
                if rep == 'SALAD':
                    require(np.array_equal(ids[ranking[:20]], scores['c20'][qi]), 'original C20 not reproduced: ' + q['qid'])
                    prefix_checks += 1
                domains = [('full', full_ix, full_ix)]
                if rep == 'SALAD':
                    domains += [('C20', c20_ix, np.arange(20))]
                    domains += [(f'K{k}', ranking[:k], ranking[:k]) for k in (5, 10, 20, 50)]
                for policy, ix, keys in domains:
                    selected_ids, rr, tt, ix = select(scores, qi, rep, ix)
                    for radius in RADII:
                        legacy = old.get((q['qid'], rep, policy, radius))
                        row, cert = record(q, qi, rep, policy, selected_ids, rr, tt, ix, geo, keys,
                                           radius, legacy, fixed303)
                        rows.append(row)
                        f.write(json.dumps(cert, ensure_ascii=False, allow_nan=False) + '\n')
            if qi % 200 == 0:
                print('ABILITY', qi, '/', len(queries), 'seconds', round(time.monotonic() - start, 1), flush=True)
    write_csv(out / 'query_results.csv', rows)
    write_csv(out / 'statistics.csv', summarize(rows, METRICS))
    fixed = []
    for rep, policy, radius in sorted({(r['representation'], r['policy'], r['radius_m']) for r in rows}):
        subset = [r for r in rows if r['fixed303'] and r['representation'] == rep and r['policy'] == policy and r['radius_m'] == radius]
        fixed.append(dict(representation=rep, policy=policy, radius_m=radius, queries=len(subset),
                          **{m: sum(r[m] for r in subset) for m in METRICS}))
    write_csv(out / 'fixed303.csv', fixed)
    partitions = []
    for rep in ('SALAD', 'Clique'):
        for radius in RADII:
            subset = [r for r in rows if r['representation'] == rep and r['policy'] == 'full' and
                      r['route'] == 'SNU' and group(r) == 'morning_four' and r['radius_m'] == radius]
            for cls in ('NO_POSITIVE', 'EQUAL_CORRECT', 'SCALAR_OPPORTUNITY', 'NONLINEAR_D_OPPORTUNITY', 'STRICT_BOTH_OBSTRUCTION'):
                partitions.append(dict(representation=rep, radius_m=radius, classification=cls,
                                       count=sum(r['classification'] == cls for r in subset), queries=len(subset)))
    write_csv(out / 'partitions.csv', partitions)
    write_json(out / 'ABILITY_RECEIPT.json', dict(rows=len(rows), legacy_rows_replayed=len(prior),
        C20_prefix_checks=prefix_checks, seconds=time.monotonic() - start, independent_audit='PENDING',
        code_sha256=sha(__file__), monotone_sha256=sha(ROOT / 'code/monotone.py')))
    print('ABILITY_COMPLETE', len(rows), flush=True)

def overlap():
    queries, scores = load_scores()
    geo = load_labels()
    meta = read_json(ROOT / 'inputs/frame_metadata.json')
    frames = meta['frames']
    official = {f"{r['location']}:{r['condition']}:{r['anchor_id']}": r
                for r in read_csv(ROOT / 'inputs/official_top20.csv') if r['rank'] == '1'}
    pred = dict(np.load(ROOT / 'inputs/predictions.npz', allow_pickle=False))
    seq = {'SNU': {'M': '04', 'D': '05', 'N': '06'}, 'Valley': {'M': '07', 'D': '08', 'N': '09'}}
    def identity(frame, bank):
        if frame['route'] == 'NSAVP':
            return ':'.join(map(str, ('NSAVP', frame['sequence'], 'locked_bank', bank)))
        return ':'.join(map(str, (frame['route'], frame['sequence'], frame['split'], frame['original_id'])))
    rows = []
    for qi, q in enumerate(queries):
        ids = scores['ids'][qi, :int(scores['lengths'][qi])]
        lookup = {int(v): i for i, v in enumerate(ids)}
        methods = {}
        for name, key in [('SALAD', 'SALAD_full_weights'), ('Clique', 'Clique_full_weights')]:
            bank = int(pred[key][5, qi, 0]); j = lookup[bank]
            methods[name] = (identity(frames[bank], bank), float(geo[qi, j]))
        if q['route'] != 'NSAVP':
            o = official[q['qid']]
            target = seq[q['route']][q['condition'].split('_to_')[1]]
            ident = ':'.join(map(str, (q['route'], target, 'database', int(o['database_id']))))
            matching = [int(v) for v in ids if identity(frames[int(v)], int(v)) == ident]
            require(len(matching) == 1, 'official candidate identity not unique')
            require(abs(float(geo[qi, lookup[matching[0]]]) - float(o['geographic_distance_m'])) < 1e-7,
                    'official geographic identity mismatch')
            methods['Official'] = (ident, float(o['geographic_distance_m']))
        for a, b in itertools.combinations(methods, 2):
            ia, da = methods[a]; ib, db = methods[b]
            for radius in RADII:
                ca, cb = da <= radius, db <= radius
                row = {k: q[k] for k in ('qid', 'route', 'condition', 'anchor', 'block')}
                row.update(pair=a + '_vs_' + b, radius_m=radius, a_correct=int(ca), b_correct=int(cb),
                    both_correct=int(ca and cb), a_only=int(ca and not cb), b_only=int(cb and not ca),
                    both_wrong=int(not ca and not cb), delta_correct=int(cb) - int(ca),
                    same_wrong_candidate=int(not ca and not cb and ia == ib),
                    a_distance_m=da, b_distance_m=db, a_identity=ia, b_identity=ib)
                rows.append(row)
    write_csv(ROOT / 'analysis/model_overlap_queries.csv', rows)
    metrics = ('both_correct', 'a_only', 'b_only', 'both_wrong', 'same_wrong_candidate', 'delta_correct')
    write_csv(ROOT / 'analysis/model_overlap.csv', summarize(rows, metrics, by=('pair', 'radius_m')))
    write_json(ROOT / 'analysis/OVERLAP_RECEIPT.json', dict(rows=len(rows), official_top1_queries=len(official),
        official_NSAVP='NOT_RUN', identity_key='route:sequence:split:original_id', code_sha256=sha(__file__)))
    print('OVERLAP_COMPLETE', len(rows), flush=True)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['ability', 'overlap'])
    a = p.parse_args()
    ability() if a.stage == 'ability' else overlap()
