"""Open target geography only after all 27 GPU prediction files are locked."""
from collections import defaultdict
from io_utils import *

def main():
    lock = read_json(ROOT / 'training/PREDICTIONS_LOCK.json')
    require(len(lock['files']) == 27 and lock['target_labels_used'] is False, 'complete prediction lock required')
    for name, digest in lock['files'].items():
        require(sha(ROOT / name) == digest, 'prediction changed before evaluation')
    queries, scores = load_scores(); geo = load_labels()
    qi_map = {q['qid']: i for i, q in enumerate(queries)}
    old = {(r['qid'], int(r['radius_m'])): r for r in read_csv(ROOT / 'analysis/query_results.csv')
           if r['representation'] == 'SALAD' and r['policy'] == 'full'}
    rows, seed_stats = [], []
    violations = []
    for filename in lock['files']:
        path = ROOT / filename; done = read_json(path.with_name('COMPLETE.json')); identity = done['identity']
        arm, seed, held = identity['arm'], identity['seed'], identity['held']
        pred = np.load(path, allow_pickle=False); fit_rows = []
        require(len(pred['qids']) == sum(q['route'] == held for q in queries), 'held route query coverage')
        for j, qid in enumerate(pred['qids']):
            qi = qi_map[str(qid)]; q = queries[qi]
            require(q['route'] == held, 'prediction escaped held route')
            ids = scores['ids'][qi, :int(scores['lengths'][qi])]; lookup = {int(b): k for k, b in enumerate(ids)}
            top = pred['top20'][j]; require(len(np.unique(top)) == 20 and all(int(b) in lookup for b in top), 'candidate domain changed')
            distances = geo[qi, [lookup[int(b)] for b in top]]
            for radius in RADII:
                baseline = old[str(qid), radius]; correct = int(distances[0] <= radius); equal = int(baseline['equal_correct'])
                upper = int(baseline['scalar_stable'] if arm == 'scalar' else baseline['M_stable'])
                if arm != 'unconstrained' and correct > upper:
                    violations.append(dict(qid=str(qid), arm=arm, seed=seed, radius_m=radius, reason='claimed score-class bound violated'))
                row = {k: q[k] for k in ('qid', 'route', 'condition', 'anchor', 'block')}
                row.update(arm=arm, seed=seed, radius_m=radius, correct=correct, baseline_correct=equal,
                    gain=correct-equal, rescue=int(correct and not equal), harm=int(equal and not correct),
                    recall5=int(np.any(distances[:5] <= radius)), recall20=int(np.any(distances <= radius)),
                    top1=int(top[0]), top1_distance_m=float(distances[0]),
                    baseline_classification=baseline['classification'], theoretical_stable_upper=upper if arm != 'unconstrained' else '',
                    scalar_impossible=int(baseline['scalar_impossible']), D_impossible=int(baseline['D_impossible']))
                require(row['rescue'] - row['harm'] == row['gain'], 'rescue/harm identity')
                rows.append(row); fit_rows.append(row)
        stats = summarize(fit_rows, ('correct', 'baseline_correct', 'gain', 'rescue', 'harm', 'recall5', 'recall20'), by=('arm', 'seed', 'radius_m'))
        for stat in stats:
            if stat['metric'] != 'correct':
                continue
            subset = [r for r in fit_rows if r['radius_m'] == stat['radius_m'] and
                      (stat['group'] == 'all_records' or stat['group'] == r['condition'] or stat['group'] == group(r)
                       or stat['group'] == 'all_cross' and group(r) != 'same')]
            seed_stats.append(dict(arm=arm, seed=seed, route=held, group=stat['group'], radius_m=stat['radius_m'],
                queries=len(subset), correct=sum(r['correct'] for r in subset), R1_percent=stat['value_percent'],
                gain_pp=sum(r['gain'] for r in subset) / len(subset) * 100,
                rescue=sum(r['rescue'] for r in subset), harm=sum(r['harm'] for r in subset)))
    write_json(ROOT / 'training/CLASS_BOUND_AUDIT.json', dict(status='PASSED' if not violations else 'FAILED',
        violations=violations, rows_checked=len(rows), target_labels_opened_after_lock=True))
    require(not violations, 'score-class violation; inspect numerical/implementation error before proceeding')
    write_csv(ROOT / 'training/query_results.csv', rows); write_csv(ROOT / 'training/seed_summary.csv', seed_stats)
    by_query = defaultdict(list)
    for row in rows:
        by_query[row['arm'], row['qid'], row['radius_m']].append(row)
    means = []
    for (arm, qid, radius), values in by_query.items():
        require({r['seed'] for r in values} == {42, 43, 44}, 'seed completeness')
        row = {k: values[0][k] for k in ('qid', 'route', 'condition', 'anchor', 'block', 'arm', 'radius_m')}
        row.update({k: float(np.mean([r[k] for r in values])) for k in ('correct', 'baseline_correct', 'gain', 'rescue', 'harm', 'recall5', 'recall20')})
        means.append(row)
    mean_stats = summarize(means, ('correct', 'baseline_correct', 'gain', 'rescue', 'harm', 'recall5', 'recall20'), by=('arm', 'radius_m'))
    write_csv(ROOT / 'training/mean_summary.csv', mean_stats)
    class_rows = []
    buckets = defaultdict(list)
    for r in rows:
        buckets[r['arm'], r['seed'], r['route'], r['radius_m'], r['baseline_classification']].append(r)
    for key, values in buckets.items():
        class_rows.append(dict(zip(('arm', 'seed', 'route', 'radius_m', 'baseline_classification'), key),
            queries=len(values), correct=sum(r['correct'] for r in values), rescue=sum(r['rescue'] for r in values), harm=sum(r['harm'] for r in values)))
    write_csv(ROOT / 'training/class_transitions.csv', class_rows)
    deletions, decisions = [], []
    blocks = read_json(ROOT / 'inputs/draws.json')['SNU']['blocks']
    for arm in ('scalar', 'monotone', 'unconstrained'):
        main = [r for r in means if r['arm'] == arm and r['route'] == 'SNU' and r['radius_m'] == 10 and group(r) == 'morning_four']
        gain = float(np.mean([r['gain'] for r in main]) * 100)
        directional = [float(np.mean([r['gain'] for r in main if r['condition'] == c]) * 100) for c in ('M_to_D', 'D_to_M', 'M_to_N', 'N_to_M')]
        deletion_values = []
        for bi, block in enumerate(blocks):
            value = float(np.mean([r['gain'] for r in main if r['anchor'] not in block]) * 100)
            deletion_values.append(value); deletions.append(dict(arm=arm, deleted_raw_block=bi, SNU_morning_gain_pp=value))
        def delta(route, name):
            vals = [r['gain'] for r in means if r['arm'] == arm and r['route'] == route and r['radius_m'] == 10 and
                    (group(r) == name or name == 'all_cross' and group(r) != 'same')]
            return float(np.mean(vals) * 100), float(sum(vals))
        dn, _ = delta('SNU', 'day_night'); valley, _ = delta('Valley', 'all_cross'); _, ns = delta('NSAVP', 'all_cross')
        checks = dict(SNU_gain_at_least_4pp=gain >= 4, at_least_three_directions=sum(v > 0 for v in directional) >= 3,
                      all_block_deletions_positive=min(deletion_values) > 0,
                      DayNight_loss_at_most_1pp=dn >= -1, Valley_loss_at_most_1pp=valley >= -1,
                      NSAVP_net_loss_at_most_one=ns >= -1)
        checks = {k: bool(v) for k, v in checks.items()}
        decisions.append(dict(arm=arm, SNU_morning_gain_pp=gain, directional_gain_pp=directional,
            DayNight_gain_pp=dn, Valley_cross_gain_pp=valley, NSAVP_net_gain_queries=ns, checks=checks,
            passed=all(checks.values())))
    write_csv(ROOT / 'training/block_deletions.csv', deletions)
    passed = [r for r in decisions if r['passed']]
    selected = max(passed, key=lambda r: r['SNU_morning_gain_pp'])['arm'] if passed else None
    write_json(ROOT / 'training/DECISION.json', dict(status='METHOD_INVESTMENT_GATE_PASSED' if passed else 'NO_ARM_PASSED_FIXED_INVESTMENT_GATE',
        arms=decisions, selected_arm=selected, seed_selection=False, project_routes='development exposed',
        further_target_driven_search=False, independent_method_validation='NOT_COMPLETED'))
    print('TRAINING_EVALUATED', len(rows), 'selected', selected, flush=True)

if __name__ == '__main__':
    main()
