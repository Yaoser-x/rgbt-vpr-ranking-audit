"""Independent rational reference audit; does not import any production module.

Every float64 input is treated as its exact binary rational value.  The
reference intersects positive-versus-negative inequalities using Fraction;
the production implementation uses an integer-lattice solver instead.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
ZERO, ONE = Fraction(0), Fraction(1)


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def rational(value):
    return Fraction.from_float(float(value))


def is_empty(interval):
    low, low_closed, high, high_closed = interval
    return low > high or (low == high and not (low_closed and high_closed))


def union_intervals(intervals):
    """Canonical exact union, retaining singletons and excluded touching points."""
    ordered = sorted((v for v in intervals if not is_empty(v)), key=lambda v: (v[0], not v[1], v[2], not v[3]))
    result = []
    for low, low_closed, high, high_closed in ordered:
        if not result:
            result.append((low, low_closed, high, high_closed))
            continue
        old_low, old_lc, old_high, old_hc = result[-1]
        if low < old_high or (low == old_high and (low_closed or old_hc)):
            if high > old_high:
                result[-1] = (old_low, old_lc, high, high_closed)
            elif high == old_high:
                result[-1] = (old_low, old_lc, old_high, old_hc or high_closed)
        else:
            result.append((low, low_closed, high, high_closed))
    return result


def serialize_intervals(intervals):
    return [dict(low=str(low), low_closed=bool(lc), high=str(high), high_closed=bool(hc))
            for low, lc, high, hc in intervals]


def reference(dr, dt, positive, optimistic=False):
    """Return exact feasible union and two-constraint empty certificates per P.

    Candidate array order itself defines stable ties.  Positive-positive
    comparisons are unnecessary: success requires some positive ahead of all
    negatives, not that a preselected positive outranks other positives.
    """
    require(len(dr) == len(dt) == len(positive), 'reference input lengths')
    require(np.isfinite(dr).all() and np.isfinite(dt).all(), 'nonfinite input')
    r = [rational(v) for v in dr]
    t = [rational(v) for v in dt]
    slopes = [a - b for a, b in zip(r, t)]
    ps = np.flatnonzero(positive).tolist()
    ns = np.flatnonzero(~np.asarray(positive, dtype=bool)).tolist()
    feasible, certificates = [], []
    for p in ps:
        low, high = ZERO, ONE
        lc = hc = True
        lower_by, upper_by = None, None
        parallel_blocker = None
        for n in ns:
            a = slopes[p] - slopes[n]
            b = t[n] - t[p]
            closed = bool(optimistic or p < n)
            if a == 0:
                if b < 0 or (b == 0 and not closed):
                    parallel_blocker = n
                    break
                continue
            bound = b / a
            if a > 0:
                if bound < high:
                    high, hc, upper_by = bound, closed, n
                elif bound == high:
                    hc = hc and closed
                    if not closed:
                        upper_by = n
            else:
                if bound > low:
                    low, lc, lower_by = bound, closed, n
                elif bound == low:
                    lc = lc and closed
                    if not closed:
                        lower_by = n
            if is_empty((low, lc, high, hc)):
                break
        interval = (low, lc, high, hc)
        if parallel_blocker is None and not is_empty(interval):
            feasible.append(interval)
        else:
            certificates.append(dict(positive_local=p, parallel_negative=parallel_blocker,
                                     lower_negative=lower_by, upper_negative=upper_by,
                                     low=str(low), low_closed=lc, high=str(high), high_closed=hc))
    return dict(intervals=union_intervals(feasible), certificates=certificates,
                positive_count=len(ps), negative_count=len(ns))


def exact_winner(dr, dt, alpha):
    scores = [alpha * rational(r) + (ONE - alpha) * rational(t) for r, t in zip(dr, dt)]
    return min(range(len(scores)), key=lambda index: (scores[index], index))


def contains(intervals, alpha):
    return any((alpha > low or (alpha == low and lc)) and (alpha < high or (alpha == high and hc))
               for low, lc, high, hc in intervals)


def self_test():
    # Stable ties can make the positive lose an otherwise feasible endpoint.
    a = reference(np.array([0., 1.]), np.array([1., 0.]), np.array([False, True]))
    require(a['intervals'] == [(ZERO, True, Fraction(1, 2), False)], 'stable endpoint')
    # A positive at each extreme leaves an interior hole.
    a = reference(np.array([0., .4, 1.]), np.array([1., .4, 0.]), np.array([True, False, True]))
    require(len(a['intervals']) == 2 and not contains(a['intervals'], Fraction(1, 2)), 'multi-positive hole')
    # A positive can tie at a singleton while stable order blocks it entirely.
    r = np.array([0., .5, 1.]); t = np.array([1., .5, 0.]); p = np.array([False, True, False])
    require(not reference(r, t, p)['intervals'], 'stable singleton blocked')
    require(reference(r, t, p, True)['intervals'] == [(Fraction(1, 2), True, Fraction(1, 2), True)], 'optimistic singleton')
    # Fully duplicate rows retain original candidate order.
    require(not reference([1., 1.], [2., 2.], [False, True])['intervals'], 'duplicate stable blocked')
    require(reference([1., 1.], [2., 2.], [False, True], True)['intervals'] == [(ZERO, True, ONE, True)], 'duplicate optimistic')
    require(not reference([1.], [2.], [False])['intervals'], 'no positive')
    require(reference([1.], [2.], [True])['intervals'] == [(ZERO, True, ONE, True)], 'no negative')
    require(union_intervals([(ZERO, True, Fraction(1, 2), False), (Fraction(1, 2), False, ONE, True)]) ==
            [(ZERO, True, Fraction(1, 2), False), (Fraction(1, 2), False, ONE, True)], 'open touching intervals')
    # Unsupported Pareto point: max can recover information linear fusion loses.
    r = np.array([0., .6, 1.]); t = np.array([1., .6, 0.]); p = np.array([False, True, False])
    require(not reference(r, t, p, True)['intervals'] and int(np.argmin(np.maximum(r, t))) == 1,
            'max recovers unsupported nondominated positive')
    # Exact endpoint feasibility must not be rounded away.
    r = np.array([1., 1.]); t = np.array([2., 1.]); p = np.array([True, False])
    require(reference(r, t, p)['intervals'] == [(ONE, True, ONE, True)], 'endpoint only')
    return 11


def frozen_inputs(package):
    """Bind IDs and position labels directly to old inputs, independently."""
    repo = PROJECT / 'rgbt-vpr-validity'
    pilot = repo / 'Paper-A/GPU-Adaptation-Pilot-v1/artifacts'
    meta_path = pilot / 'prepared_v1/data.json'
    meta = read_json(meta_path)
    queries = meta['queries']
    with np.load(pilot / 'prepared_v1/evaluation.npz', allow_pickle=False) as source:
        geo, dr, dt, c20 = (source[key].copy() for key in ('geo', 'dr', 'dt', 'c20'))
    score_path = package / 'prepared/score_only.npz'
    with np.load(score_path, allow_pickle=False) as source:
        scores = {key: source[key].copy() for key in source.files}
    lock = read_json(package / 'provenance/PREDICTIONS_LOCK.json')
    require(sha(score_path) == lock['score_only_sha256'], 'score hash is not locked')
    require(sha(package / 'predictions.npz') == lock['prediction_sha256'], 'prediction hash is not locked')
    require(sha(package / 'PROTOCOL.json') == lock['protocol_sha256'], 'protocol hash is not locked')
    require(set(scores) == {'dr_s', 'dt_s', 'dr_c', 'dt_c', 'ids', 'lengths', 'c20', 'qids'}, 'score-only key whitelist')
    require(scores['qids'].tolist() == [q['qid'] for q in queries], 'metadata order or qid changed')
    require(np.array_equal(scores['c20'], c20), 'old original C20 binding changed')
    require(np.array_equal(scores['dr_s'], dr, equal_nan=True) and np.array_equal(scores['dt_s'], dt, equal_nan=True),
            'original SALAD scores changed')
    require(len(queries) == 2298 and len({q['qid'] for q in queries}) == 2298, 'cohort or qid uniqueness')
    for qi, query in enumerate(queries):
        gallery = np.array(query['gallery_indices'], dtype=np.int64)
        count = len(gallery)
        require(count == int(scores['lengths'][qi]) and count >= 20 and len(set(gallery)) == count,
                'gallery count/duplicates')
        require(np.array_equal(gallery, scores['ids'][qi, :count]), 'gallery identity/order changed')
        require(set(c20[qi]).issubset(set(gallery)) and len(set(c20[qi])) == 20, 'C20 identity/budget')
        require(np.isfinite(geo[qi, :count]).all(), 'nonfinite real geography')
        for key in ('dr_s', 'dt_s', 'dr_c', 'dt_c'):
            require(scores[key].dtype == np.float64 and np.isfinite(scores[key][qi, :count]).all(), 'real score finite float64')
    return meta, geo, scores


def independently_reconstruct_clique(scores, queries):
    """Recompute all frozen Clique squared distances, not only top-20 winners."""
    from scipy.spatial.distance import cdist
    directory = PROJECT / 'rgbt-vpr-validity/Paper-A/GPU-Adaptation-Pilot-v1/artifacts/clique_v1'
    arrays = {name: np.load(directory / (name + '.npy'), mmap_mode='r', allow_pickle=False)
              for name in ('rgb', 'thermal')}
    groups = defaultdict(list)
    for qi, query in enumerate(queries):
        groups[tuple(query['gallery_indices'])].append((qi, int(query['query_index'])))
    count = 0
    for gallery, members in groups.items():
        qindices = np.array([i for _, i in members], dtype=np.int64)
        for name, score in (('rgb', 'dr_c'), ('thermal', 'dt_c')):
            reference_distance = cdist(np.asarray(arrays[name][qindices], dtype=np.float64),
                                       np.asarray(arrays[name][list(gallery)], dtype=np.float64), metric='sqeuclidean')
            for j, (qi, _) in enumerate(members):
                require(np.array_equal(reference_distance[j], scores[score][qi, :len(gallery)]),
                        'Clique full score recomputation mismatch: ' + queries[qi]['qid'])
        count += len(members)
        print('AUDIT_CLIQUE_DISTANCE', count, '/', len(queries), flush=True)
    return count


def parse_bound(value):
    if isinstance(value, str):
        return Fraction(value)
    if isinstance(value, list) and len(value) == 2:
        return Fraction(int(value[0]), int(value[1]))
    if isinstance(value, dict):
        return Fraction(int(value.get('numerator', value.get('num'))), int(value.get('denominator', value.get('den'))))
    return Fraction(value)


def parse_production_intervals(values):
    result = []
    for value in values:
        if isinstance(value, dict):
            if 'lower' in value and 'upper' in value:
                lower, upper = value['lower'], value['upper']
                result.append((parse_bound(lower), bool(lower['closed']), parse_bound(upper), bool(upper['closed'])))
                continue
            result.append((parse_bound(value.get('low', value.get('lo'))),
                           bool(value.get('low_closed', value.get('lo_closed'))),
                           parse_bound(value.get('high', value.get('hi'))),
                           bool(value.get('high_closed', value.get('hi_closed')))))
        else:
            require(len(value) == 4, 'unexpected production interval schema')
            result.append((parse_bound(value[0]), bool(value[1]), parse_bound(value[2]), bool(value[3])))
    return result


def audit_outputs(package):
    start = time.monotonic()
    require(read_json(package / 'audit/INPUTS_REFERENCE.json')['passed'], 'independent input audit required')
    meta, geo, scores = frozen_inputs(package)
    queries = meta['queries']
    query_map = {query['qid']: (qi, query) for qi, query in enumerate(queries)}
    with np.load(package / 'predictions.npz', allow_pickle=False) as source:
        predictions = {key: source[key].copy() for key in source.files}
    require(predictions['qids'].tolist() == [q['qid'] for q in queries], 'prediction query identity')
    counts = Counter()
    numerical = []
    seen = set()
    audit_file = package / 'audit/FRACTION_REFERENCE.jsonl'
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    with (package / 'analysis/intervals.jsonl').open(encoding='utf-8') as production, audit_file.open('w', encoding='utf-8') as out:
        for line_number, line in enumerate(production, 1):
            result = json.loads(line)
            representation, policy, qid = (result[key] for key in ('representation', 'policy', 'qid'))
            key = (representation, policy, qid)
            require(key not in seen, 'duplicate interval result')
            seen.add(key)
            require(representation in ('SALAD', 'Clique') and policy in ('full', 'C20'), 'unexpected analysis condition')
            require(not (representation == 'Clique' and policy == 'C20'), 'unplanned Clique C20 condition')
            qi, query = query_map[qid]
            n = int(scores['lengths'][qi])
            gallery = scores['ids'][qi, :n]
            all_local = {int(bank): i for i, bank in enumerate(gallery)}
            candidates = gallery if policy == 'full' else scores['c20'][qi]
            indices = np.asarray([all_local[int(bank)] for bank in candidates], dtype=np.int64)
            require(list(map(int, candidates)) == result['candidate_bank_indices'], 'interval candidate identity/order')
            branch = 's' if representation == 'SALAD' else 'c'
            r, t = scores['dr_' + branch][qi, indices], scores['dt_' + branch][qi, indices]
            distances = geo[qi, indices]
            positive = distances <= 10
            stable = reference(r, t, positive)
            optimistic = reference(r, t, positive, True)
            solution, grid = result['solution'], result['grid']
            require(parse_production_intervals(solution['attainable_intervals']) == stable['intervals'], 'stable interval union mismatch: ' + repr(key))
            require(parse_production_intervals(solution['optimistic_intervals']) == optimistic['intervals'], 'optimistic interval union mismatch: ' + repr(key))
            require(bool(solution['attainable']) == bool(stable['intervals']) and
                    bool(solution['optimistic']) == bool(optimistic['intervals']), 'continuous feasibility bool mismatch')
            require(int(solution['positive_count']) == int(positive.sum()) and
                    int(solution['negative_count']) == int((~positive).sum()), 'positive/negative count mismatch')
            width = sum((high - low for low, _, high, _ in stable['intervals']), ZERO)
            require(parse_bound(solution['width_exact']) == width, 'interval width exact mismatch')
            shape = 'empty' if not stable['intervals'] else ('disconnected' if len(stable['intervals']) > 1 else
                    ('singleton' if stable['intervals'][0][0] == stable['intervals'][0][2] else 'interval'))
            require(solution['shape'] == shape, 'union topology mismatch')
            ps, ns = np.flatnonzero(positive), np.flatnonzero(~positive)
            dominated, strict_both = [], []
            for p in ps:
                dominated.append(bool(np.any((r[ns] <= r[p]) & (t[ns] <= t[p]) & ((r[ns] < r[p]) | (t[ns] < t[p])))))
                strict_both.append(bool(np.any((r[ns] < r[p]) & (t[ns] < t[p]))))
            every_dominated = bool(len(ps) and all(dominated))
            every_strict = bool(len(ps) and all(strict_both))
            undominated = bool(len(ps) and not all(dominated))
            require(bool(solution['every_positive_weakly_negative_dominated']) == every_dominated and
                    bool(solution['every_positive_strictly_both_negative_dominated']) == every_strict and
                    bool(solution['undominated_positive_exists']) == undominated, 'Pareto subtype mismatch')
            exact_grid = [contains(stable['intervals'], Fraction(j, 10)) for j in range(11)]
            require(exact_grid == grid['correct'] and int(grid['denominator']) == 10 and
                    [j for j, value in enumerate(exact_grid) if value] == grid['correct_numerators'], 'exact eleven-point flags mismatch')
            require(bool(grid['attainable']) == any(exact_grid), 'exact grid oracle bool mismatch')
            require(bool(stable['intervals']) or not any(exact_grid), 'continuous oracle lower than exact grid')
            require(len(solution['certificates']) == len(ps), 'production positive certificate coverage')
            if len(ps) and not stable['intervals']:
                require(len(stable['certificates']) == len(ps), 'independent impossible positive certificate coverage')
            if len(ps) and not optimistic['intervals']:
                require(len(optimistic['certificates']) == len(ps), 'independent optimistic impossible certificate coverage')
            prefix = representation + '_' + policy
            maximum_order = np.argsort(np.maximum(r, t), kind='stable')[:20]
            require(np.array_equal(candidates[maximum_order], predictions[prefix + '_max'][qi]), 'max locked ranking mismatch')
            equal_correct = bool(positive[np.argsort(.5 * (r + t), kind='stable')[0]])
            max_correct = bool(positive[maximum_order[0]])
            rescue, harm = int(not equal_correct and max_correct), int(equal_correct and not max_correct)
            require(rescue - harm == int(max_correct) - int(equal_correct), 'rescue-harm identity')
            float_grid = []
            binary_alpha_grid = []
            for j, alpha in enumerate(predictions['alphas']):
                order = np.argsort(alpha * r + (1 - alpha) * t, kind='stable')[:20]
                require(np.array_equal(candidates[order], predictions[prefix + '_weights'][j, qi]), 'float grid locked ranking mismatch')
                actual = bool(positive[order[0]])
                float_grid.append(actual)
                exact_binary = contains(stable['intervals'], rational(alpha))
                binary_alpha_grid.append(exact_binary)
                if actual != exact_grid[j] or actual != exact_binary:
                    numerical.append(dict(representation=representation, policy=policy, qid=qid, alpha_numerator=j,
                                          float_grid=actual, exact_rational_tenth=exact_grid[j],
                                          exact_binary_alpha=exact_binary))
                if policy == 'C20':
                    require(set(map(int, candidates[order])) == set(map(int, candidates)), 'C20 changed candidate budget')
                    require(bool(positive[order].any()) == bool(positive.any()), 'C20 R20 changed')
            if policy == 'C20':
                require(set(map(int, candidates[maximum_order])) == set(map(int, candidates)), 'C20 max budget changed')
                require(bool(positive[maximum_order].any()) == bool(positive.any()), 'C20 max R20 changed')
            require(float_grid[5] == equal_correct, 'half weight prediction differs from direct branch mean')
            category = 'NO_POSITIVE' if not len(ps) else ('STABLE_ATTAINABLE' if stable['intervals'] else
                       ('TIE_RULE_ONLY' if optimistic['intervals'] else
                        ('ALL_POSITIVES_DOMINATED' if every_dominated else 'UNSUPPORTED_UNDOMINATED_POSITIVE')))
            counts[(prefix, 'queries')] += 1
            for label, value in [('stable_attainable', bool(stable['intervals'])), ('optimistic_attainable', bool(optimistic['intervals'])),
                                 ('exact_grid_attainable', any(exact_grid)), ('float_grid_attainable', any(float_grid)),
                                 ('equal_correct', equal_correct), ('max_correct', max_correct), ('max_rescues', rescue), ('max_harms', harm),
                                 ('max_beyond_optimistic', max_correct and not optimistic['intervals']),
                                 ('impossible_with_positive', len(ps) and not stable['intervals'])]:
                counts[(prefix, label)] += int(value)
            out.write(json.dumps(dict(representation=representation, policy=policy, qid=qid,
                stable=serialize_intervals(stable['intervals']), optimistic=serialize_intervals(optimistic['intervals']),
                stable_impossible_certificates=stable['certificates'] if not stable['intervals'] else [],
                optimistic_impossible_certificates=optimistic['certificates'] if not optimistic['intervals'] else [],
                independently_derived_class=category, exact_grid=exact_grid, float_grid=float_grid,
                equal_correct=equal_correct, max_correct=max_correct, max_rescue=rescue, max_harm=harm),
                ensure_ascii=False, allow_nan=False) + '\n')
            if line_number % 300 == 0:
                print('AUDIT_FRACTION', line_number, '/ 6894', 'elapsed_s', round(time.monotonic() - start, 1), flush=True)
    require(len(seen) == 6894, 'incomplete interval output: expected three conditions x2298')
    write_json(package / 'audit/NUMERICAL_REFERENCE.json', numerical)
    receipt = {
        'passed': True, 'reference_self_test_cases': self_test(), 'query_policy_conditions_checked': len(seen),
        'all_stable_and_optimistic_unions_independently_recomputed': True,
        'impossible_queries_proven_from_exact_binary64_rational_inequalities': True,
        'independent_arithmetic': 'fractions.Fraction.from_float; no production solver/source module imports',
        'actual_float_predictions_independently_reconstructed_without_geography': True,
        'exact_grid_definition': 'j/10, j=0..10; binary64 alpha and actual float score rounding separately checked',
        'numerical_grid_flag_disagreements': len(numerical),
        'counts': {prefix: {label: value for (p, label), value in counts.items() if p == prefix}
                   for prefix in ('SALAD_full', 'Clique_full', 'SALAD_C20')},
        'reference_file_sha256': sha(audit_file), 'analysis_interval_sha256': sha(package / 'analysis/intervals.jsonl'),
        'prediction_sha256': sha(package / 'predictions.npz'), 'score_sha256': sha(package / 'prepared/score_only.npz'),
        'audit_source_sha256': sha(__file__), 'elapsed_s': time.monotonic() - start,
    }
    write_json(package / 'audit/INDEPENDENT_AUDIT.json', receipt)
    print('AUDIT_FRACTION_PASSED', len(seen), 'elapsed_s', round(receipt['elapsed_s'], 2), flush=True)
    return receipt


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as source:
        return list(csv.DictReader(source))


def same_group(condition, group):
    same = condition in ('M_to_M', 'D_to_D', 'N_to_N')
    if group == 'all_records':
        return True
    if group == 'same':
        return same
    if group == 'all_cross':
        return not same
    if group == 'morning_four':
        return condition in ('M_to_D', 'M_to_N', 'D_to_M', 'N_to_M')
    if group == 'day_night':
        return condition in ('D_to_N', 'N_to_D')
    return condition == group


def audit_tables(package):
    """Independent aggregation and the exact old spatial resampling draws."""
    start = time.monotonic()
    reference_receipt = read_json(package / 'audit/INDEPENDENT_AUDIT.json')
    require(reference_receipt['passed'], 'reference constraints must pass before tables')
    meta, geo, scores = frozen_inputs(package)
    queries = meta['queries']
    lookup = {q['qid']: (i, q) for i, q in enumerate(queries)}
    with (package / 'audit/FRACTION_REFERENCE.jsonl').open(encoding='utf-8') as source:
        reference_rows = {(row['representation'], row['policy'], row['qid']): row
                          for row in map(json.loads, source)}
    observed = csv_rows(package / 'analysis/query_results.csv')
    require(len(observed) == len(reference_rows) == 6894, 'query result row budget')
    seen = set()
    with np.load(package / 'predictions.npz', allow_pickle=False) as source:
        predictions = {key: source[key].copy() for key in source.files}
    for row in observed:
        key = (row['representation'], row['policy'], row['qid'])
        require(key not in seen and key in reference_rows, 'query table duplicate/binding')
        seen.add(key)
        ref = reference_rows[key]
        qi, query = lookup[row['qid']]
        require(all(str(query[field]) == row[field] for field in ('route', 'condition', 'anchor', 'block')), 'query metadata column')
        n = int(scores['lengths'][qi])
        gids = scores['ids'][qi, :n]
        bank_to_local = {int(bank): local for local, bank in enumerate(gids)}
        candidates = gids if row['policy'] == 'full' else scores['c20'][qi]
        local = np.array([bank_to_local[int(bank)] for bank in candidates], dtype=np.int64)
        branch = 's' if row['representation'] == 'SALAD' else 'c'
        r, t = scores['dr_' + branch][qi, local], scores['dt_' + branch][qi, local]
        d = geo[qi, local]
        equal_order = np.argsort(.5 * (r + t), kind='stable')[:20]
        max_order = np.argsort(np.maximum(r, t), kind='stable')[:20]
        positive = d <= 10
        equal, maximum = bool(positive[equal_order[0]]), bool(positive[max_order[0]])
        equal_distance, max_distance = float(d[equal_order[0]]), float(d[max_order[0]])
        stable, optimistic = bool(ref['stable']), bool(ref['optimistic'])
        c20local = np.array([bank_to_local[int(bank)] for bank in scores['c20'][qi]], dtype=np.int64)
        c20positive = geo[qi, c20local] <= 10
        # A common original-SALAD C20 availability reference; each row's own
        # representation/policy mean score defines whether that row is wrong.
        eligible = bool(c20positive.any() and not equal)
        expected = {
            'gallery_size': len(candidates), 'positive_count': int(positive.sum()),
            'equal_correct': int(equal), 'max_correct': int(maximum),
            'equal_top1': int(candidates[equal_order[0]]), 'max_top1': int(candidates[max_order[0]]),
            'top1_changed': int(candidates[equal_order[0]] != candidates[max_order[0]]),
            'rescue': int(not equal and maximum), 'harm': int(equal and not maximum),
            'max_minus_equal': int(maximum) - int(equal),
            'two_top1_oracle': int(positive[np.argmin(r)] or positive[np.argmin(t)]),
            'grid11_oracle': int(any(ref['exact_grid'])), 'continuous_oracle': int(stable),
            'optimistic_oracle': int(optimistic), 'linear_headroom': int(stable) - int(equal),
            'linear_impossible': int(bool(positive.any()) and not optimistic),
            'tie_only_blocked': int(not stable and optimistic), 'salad_c20_available': int(c20positive.any()),
            'max_rescue_beyond_linear': int(maximum and not optimistic),
            'new_far_harm': int(equal and not maximum and max_distance > 25),
            'equal_near_error': int(10 < equal_distance <= 25), 'max_near_error': int(10 < max_distance <= 25),
            'equal_far_error': int(equal_distance > 25), 'max_far_error': int(max_distance > 25),
        }
        for k in (5, 20):
            expected['equal_r' + str(k)] = int(positive[equal_order[:k]].any())
            expected['max_r' + str(k)] = int(positive[max_order[:k]].any())
        for j in range(11):
            expected[f'weight_{j:02d}_correct'] = int(ref['float_grid'][j])
            expected[f'exact_weight_{j:02d}_correct'] = int(ref['exact_grid'][j])
        for field, value in expected.items():
            require(int(row[field]) == value, 'query table value mismatch: ' + repr(key) + ' ' + field)
        require(float(row['equal_distance_m']) == equal_distance and float(row['max_distance_m']) == max_distance,
                'query winner geographic distance')
        if row['policy'] == 'C20':
            require(row['equal_r20'] == row['max_r20'], 'query fixed-C20 recall20')
        require(int(row['rescue']) - int(row['harm']) == int(row['max_minus_equal']), 'table rescue-harm identity')
        require(int(row['eligible_C_error_C20']) == int(eligible), 'row-specific mean error/common C20 eligibility definition')
        require(int(row['eligible_linear_impossible']) == int(eligible and not optimistic), 'eligible optimistic-impossible definition')
    raw_blocks, picks = {}, {}
    draw_dir = PROJECT / 'rgbt-vpr-validity/results/runs/20260918T_stage4a_location_time_interaction'
    for route in ('SNU', 'Valley'):
        with np.load(draw_dir / ('bootstrap_block_draws_' + route + '.npz'), allow_pickle=True) as source:
            raw_blocks[route] = [{str(int(anchor)) for anchor in block} for block in source['raw_block_anchor_ids']]
            picks[route] = source['block_picks'].copy()
        require(picks[route].shape == (2000, len(raw_blocks[route])), 'old raw draw shape')
    raw_blocks['NSAVP'] = [{str(q['anchor']) for q in queries if q['route'] == 'NSAVP' and int(q['block']) == block}
                           for block in range(5)]
    ns_draw = PROJECT / 'rgbt-vpr-validity/Paper-A/External-Transfer-NSAVP-v1/artifacts/run_20261017_v1/bootstrap_draws.json'
    picks['NSAVP'] = np.asarray(read_json(ns_draw), dtype=np.int64)
    require(picks['NSAVP'].shape == (2000, 5), 'NS original draw shape')
    grouped = defaultdict(list)
    for row in observed:
        grouped[(row['representation'], row['policy'], row['route'])].append(row)

    def select(row):
        return [value for value in grouped[(row['representation'], row['policy'], row['route'])]
                if same_group(value['condition'], row['group'])]

    def aggregate(values, field, route, denominator=None):
        numerator = np.array([sum(int(row[field]) for row in values if row['anchor'] in block)
                              for block in raw_blocks[route]], dtype=np.float64)
        divisor = np.array([sum(int(row[denominator]) if denominator else 1 for row in values if row['anchor'] in block)
                            for block in raw_blocks[route]], dtype=np.float64)
        require(sum(len([r for r in values if r['anchor'] in b]) for b in raw_blocks[route]) == len(values), 'raw block membership coverage')
        total_num, total_den = float(numerator.sum()), float(divisor.sum())
        if not total_den:
            return int(total_num), None, None
        draw_num, draw_den = numerator[picks[route]].sum(axis=1), divisor[picks[route]].sum(axis=1)
        usable = draw_den > 0
        interval = np.quantile(draw_num[usable] / draw_den[usable] * 100, [.025, .975]) if usable.any() else None
        return int(total_num), total_num / total_den * 100, interval

    def match_float(value, expected, label):
        if expected is None:
            require(value == '', 'expected missing numeric field: ' + label)
        else:
            require(abs(float(value) - float(expected)) < 1e-9, 'aggregate numeric mismatch: ' + label)

    stat_rows = csv_rows(package / 'analysis/statistics.csv')
    for row in stat_rows:
        values = select(row)
        require(int(row['queries']) == len(values), 'statistics group size')
        if row['metric'] == 'linear_impossible_fraction_of_C_error_C20':
            field, denominator = 'eligible_linear_impossible', 'eligible_C_error_C20'
        else:
            field, denominator = row['metric'], None
        count, point, interval = aggregate(values, field, row['route'], denominator)
        require(int(row['count']) == count, 'statistics count')
        match_float(row['value_percent'], point, row['metric'])
        match_float(row['ci95_low'], None if interval is None else interval[0], row['metric'] + ' low')
        match_float(row['ci95_high'], None if interval is None else interval[1], row['metric'] + ' high')
    weight_rows = csv_rows(package / 'analysis/fixed_weights.csv')
    for row in weight_rows:
        values = select(row)
        field = f"weight_{int(row['alpha_numerator']):02d}_correct"
        count, point, interval = aggregate(values, field, row['route'])
        require(int(row['alpha_denominator']) == 10 and int(row['queries']) == len(values) and int(row['correct']) == count,
                'weight result identity/count')
        match_float(row['R1_percent'], point, 'fixed weight point')
        match_float(row['ci95_low'], interval[0], 'fixed weight low')
        match_float(row['ci95_high'], interval[1], 'fixed weight high')
    deletion_rows = csv_rows(package / 'analysis/block_deletions.csv')
    for row in deletion_rows:
        values = select(row)
        block = raw_blocks[row['route']][int(row['deleted_raw_block'])]
        remaining = [value for value in values if value['anchor'] not in block]
        deleted_anchors = {value['anchor'] for value in values if value['anchor'] in block}
        require(int(row['deleted_anchors']) == len(deleted_anchors) and len(remaining) > 0, 'deleted raw block identity')
        expected = np.mean([int(value[row['metric']]) for value in remaining]) * 100
        match_float(row['value_percent'], expected, 'raw deletion')
    direction_rows = csv_rows(package / 'analysis/direction_summary.csv')
    for row in direction_rows:
        values = [value for value in grouped[(row['representation'], row['policy'], row['route'])]
                  if value['condition'] == row['condition']]
        require(int(row['queries']) == len(values), 'direction count')
        for field in row:
            if field not in ('representation', 'policy', 'route', 'condition', 'queries'):
                require(int(row[field]) == sum(int(value[field]) for value in values), 'direction metric sum')
    class_rows = csv_rows(package / 'analysis/classifications.csv')
    for row in class_rows:
        values = select(row)
        require(int(row['queries']) == len(values) and int(row['count']) == sum(value['classification'] == row['classification'] for value in values),
                'classification group count')
    ratio_rows = csv_rows(package / 'analysis/ratio_draw_validity.csv')
    require(len(ratio_rows) == sum(row['metric'] == 'linear_impossible_fraction_of_C_error_C20' for row in stat_rows),
            'ratio draw validity membership coverage')
    for row in ratio_rows:
        values = select(row)
        route = row['route']
        denominators = np.array([sum(int(value['eligible_C_error_C20']) for value in values if value['anchor'] in block)
                                 for block in raw_blocks[route]], dtype=np.int64)
        draw_denominators = denominators[picks[route]].sum(axis=1)
        valid = int(np.count_nonzero(draw_denominators))
        require(int(row['original_eligible_records']) == int(denominators.sum()) and
                int(row['valid_bootstrap_draws']) == valid and
                int(row['undefined_zero_denominator_draws']) == 2000 - valid and
                int(row['total_saved_draws']) == 2000, 'conditional ratio missing/valid draw counts')
    protected = read_json(package / 'provenance/protected_before.json')
    for relative, expected in protected.items():
        require(sha(PROJECT / relative) == expected, 'protected source changed: ' + relative)
    receipt = dict(passed=True, query_rows_independently_bound=len(observed), statistics_rows_recomputed=len(stat_rows),
        fixed_weight_rows_recomputed=len(weight_rows), deletion_rows_recomputed=len(deletion_rows),
        direction_rows_recomputed=len(direction_rows), classification_rows_recomputed=len(class_rows),
        conditional_ratio_validity_rows_recomputed=len(ratio_rows),
        old_2000_spatial_draws_reused=True, bootstrap_pairing='directions/methods/metrics share old anchor block picks',
        ratio_ci='resample eligible numerators and denominators with the same old blocks; do not average percentages',
        protected_sources_rehashed=len(protected), audit_source_sha256=sha(__file__), elapsed_s=time.monotonic() - start)
    write_json(package / 'audit/TABLES_REFERENCE.json', receipt)
    # Keep the independently executed Fraction proof receipt intact inside the
    # final summary; the table audit never substitutes for that recomputation.
    fraction_receipt = reference_receipt.get('fraction_reference', reference_receipt)
    inputs_receipt = read_json(package / 'audit/INPUTS_REFERENCE.json')
    fuzz_receipt = read_json(package / 'audit/REFERENCE_FUZZ.json')
    write_json(package / 'audit/FRACTION_REFERENCE_RECEIPT.json', fraction_receipt)
    write_json(package / 'audit/INDEPENDENT_AUDIT.json', {
        'passed': bool(fraction_receipt['passed'] and inputs_receipt['passed'] and receipt['passed'] and fuzz_receipt['passed']),
        'inputs_reference': inputs_receipt, 'fraction_reference': fraction_receipt,
        'tables_reference': receipt, 'synthetic_reference': fuzz_receipt,
        'production_modules_imported': False,
        'audit_source_sha256': sha(__file__),
        'report_review': 'DECISION_REPORT and MANUSCRIPT_ADDENDUM read; numeric/claim boundary review conveyed to root',
    })
    print('AUDIT_TABLES_PASSED', json.dumps(receipt), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--check-tables', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps({'passed': True, 'reference_cases': self_test()}))
    elif args.check_inputs:
        start = time.monotonic()
        meta, _, scores = frozen_inputs(ROOT)
        count = independently_reconstruct_clique(scores, meta['queries'])
        write_json(ROOT / 'audit/INPUTS_REFERENCE.json', {
            'passed': True, 'queries': count, 'all_Clique_full_scores_bitwise_recomputed': True,
            'SALAD_full_scores_equal_frozen_source': True,
            'score_only_sha256': sha(ROOT / 'prepared/score_only.npz'),
            'prediction_sha256': sha(ROOT / 'predictions.npz'),
            'elapsed_s': time.monotonic() - start,
            'audit_source_sha256': sha(__file__),
        })
        print('AUDIT_INPUTS_PASSED', count, flush=True)
    elif args.check_tables:
        audit_tables(ROOT)
    else:
        audit_outputs(ROOT)


if __name__ == '__main__':
    main()
