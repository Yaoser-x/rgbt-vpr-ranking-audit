"""Acquisition-contrast replay; frozen predictions and existing spatial draws only."""
from pathlib import Path
from collections import Counter
import csv, hashlib, json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT/'inputs/prepared'
EVAL = ROOT/'inputs/evaluation'
DRAW = ROOT/'inputs/draws'
NEAR = ROOT/'inputs/near'
METHODS = ('C', 'Clique_C', 'F_C', 'M_C')
RADII = (5,10,15,20,25)
MORNING = ('M_to_D','M_to_N','D_to_M','N_to_M')
DN = ('D_to_N','N_to_D')
CONDITIONS = tuple(f'{a}_to_{b}' for a in 'MDN' for b in 'MDN')
def verify_sources():
    records = read_json(ROOT/'PORTABLE_MEMBERS.json')
    for name, digest in records.items():
        require(sha(ROOT/name) == digest, 'changed portable member: '+name)
    return len(records)

def require(value, message):
    if not value:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def json_out(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')

def csv_out(path, rows):
    rows = list(rows)
    require(bool(rows), 'empty output: ' + str(path))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def interval(values):
    require(np.isfinite(values).all(), 'nonfinite bootstrap output')
    return [float(v) for v in np.quantile(values, [.025, .975])]

def analyze():
    verify_sources()
    require(read_json(NEAR / 'artifacts/audit_v1/AUDIT.json')['passed'], 'input audit not passed')
    meta = read_json(PREP / 'data.json')
    queries = meta['queries']
    q_lookup = {q['qid']: (i, q) for i, q in enumerate(queries)}
    require(len(q_lookup) == len(queries) == 2298, 'metadata query identity collision')
    with np.load(PREP / 'evaluation.npz', allow_pickle=False) as data:
        geo = data['geo'].copy()
        dr, dt = data['dr'].copy(), data['dt'].copy()
    with np.load(EVAL / 'rankings_top20.npz', allow_pickle=False) as data:
        rankings = data['indices'].copy()
        rank_methods = list(data['methods'])
        rank_qids = list(data['qids'])
    require(rank_qids == [q['qid'] for q in queries], 'ranking metadata order mismatch')
    rows = read_csv(EVAL / 'query_results.csv')
    source = {(r['qid'], r['method']): r for r in rows}
    require(len(source) == len(rows), 'duplicate frozen query-method rows')
    radius_input = read_csv(EVAL / 'radius_query_results.csv')
    source_radius = {(r['qid'], r['method'], int(r['radius_m'])): r for r in radius_input}
    require(len(source_radius) == len(radius_input), 'duplicate frozen query-method-radius rows')
    result = {}
    coverage = {}
    distances = {}
    query_rows = []
    diagnostic = []
    for qi, query in enumerate(queries):
        gids = np.asarray(query['gallery_indices'], np.int64)
        gd = geo[qi, :len(gids)]
        require(np.isfinite(gd).all() and (gd >= 0).all(), 'invalid published geographic distance')
        ctop = gids[np.argsort(.5 * (dr[qi, :len(gids)] + dt[qi, :len(gids)]), kind='stable')[0]]
        require(int(ctop) == int(source[query['qid'], 'C']['top1_bank_index']), 'original C top1 mismatch')
        for method in (*METHODS, 'RGB', 'Thermal'):
            frozen = source[query['qid'], method]
            bank = int(frozen['top1_bank_index'])
            indices = np.flatnonzero(gids == bank)
            require(len(indices) == 1, 'top1 not exactly once in gallery')
            distance = float(gd[indices[0]])
            require(abs(distance - float(frozen['top1_distance_m'])) <= 1e-11, 'top1 distance mismatch')
            require(int(rankings[qi, rank_methods.index(method), 0]) == bank, 'saved top20/top1 mismatch')
            distances[query['qid'], method] = distance
            if query['route'] in ('SNU', 'Valley'):
                band = 'le10' if distance <= 10 else '10to25' if distance <= 25 else 'gt25'
                diagnostic.append({k: query[k] for k in ('qid', 'route', 'condition', 'anchor', 'block')} |
                                  {'method': method, 'top1_distance_m': distance, 'band': band,
                                   'r_at_20_10m': int(frozen['r_at_20'])})
            if query['route'] not in ('SNU', 'Valley') or method not in METHODS:
                continue
            for radius in RADII:
                good = int(distance <= radius)
                can = int(np.min(gd) <= radius)
                key = (query['qid'], method, radius)
                require(int(source_radius[key]['correct']) == good, 'frozen radius correctness mismatch')
                require(int(source_radius[key]['top1_bank_index']) == bank, 'radius prediction changed')
                result[query['route'], query['condition'], query['anchor'], method, radius] = good
                coverage[query['route'], query['condition'], query['anchor'], radius] = can
                query_rows.append({k: query[k] for k in ('qid', 'route', 'condition', 'anchor', 'block')} |
                                  {'method': method, 'radius_m': radius, 'correct': good, 'available_positive': can,
                                   'top1_bank_index': bank, 'top1_distance_m': distance})
    require(len(query_rows) == 1998 * 4 * 5, 'incomplete 1998-row frozen cohort')
    anchors, raw_blocks, picks, weights = {}, {}, {}, {}
    block_members = []
    for route, expected_count in (('SNU', 133), ('Valley', 89)):
        anchors[route] = sorted({q['anchor'] for q in queries if q['route'] == route}, key=int)
        require(len(anchors[route]) == expected_count, 'anchor cohort mismatch')
        raw_blocks[route] = read_json(DRAW / f'raw_blocks_{route}.json')
        picks[route] = np.load(DRAW / f'block_picks_{route}.npz', allow_pickle=False)['picks']
        require(picks[route].shape == (2000, len(raw_blocks[route])), 'unexpected saved draw shape')
        flattened = [a for block in raw_blocks[route] for a in block]
        require(sorted(flattened, key=int) == anchors[route] and len(set(flattened)) == expected_count, 'raw block coverage mismatch')
        require(picks[route].min() >= 0 and picks[route].max() < len(raw_blocks[route]), 'invalid block pick')
        multiplicity = np.zeros((2000, expected_count), np.float64)
        ai = {a: i for i, a in enumerate(anchors[route])}
        for bi, block in enumerate(raw_blocks[route]):
            count = np.sum(picks[route] == bi, axis=1)
            for anchor in block:
                multiplicity[:, ai[anchor]] = count
                block_members.append({'route': route, 'raw_block': bi, 'anchor': anchor, 'anchors_in_block': len(block)})
        weights[route] = multiplicity / multiplicity.sum(axis=1, keepdims=True)

    direction_rows, estimates, direct, deletion = [], [], [], []
    route_vectors = {}
    route_draws = {}
    def values(route, condition, method, radius):
        return np.array([result[route, condition, a, method, radius] for a in anchors[route]], np.float64)
    for method in METHODS:
        for radius in RADII:
            for route in ('SNU', 'Valley'):
                losses = {}
                for condition in CONDITIONS:
                    v = values(route, condition, method, radius)
                    reference = condition[-1] + '_to_' + condition[-1]
                    ref = values(route, reference, method, radius)
                    loss = (ref - v) * 100
                    losses[condition] = loss
                    ci = interval(np.clip(weights[route] @ v * 100, 0, 100))
                    direction_rows.append({'route': route, 'condition': condition, 'method': method, 'radius_m': radius,
                        'anchors': len(v), 'correct': int(v.sum()), 'R_percent': float(v.mean()*100),
                        'ci95_low': ci[0], 'ci95_high': ci[1],
                        'available_positive': sum(coverage[route, condition, a, radius] for a in anchors[route]),
                        'same_gallery_reference': reference, 'reference_R_percent': float(ref.mean()*100),
                        'loss_pp': float(loss.mean())})
                vectors = {'I4': np.mean([losses[d] for d in MORNING], axis=0),
                           'IDN': np.mean([losses[d] for d in DN], axis=0)}
                vectors['J'] = vectors['I4'] - vectors['IDN']
                vectors['I6'] = (2*vectors['I4'] + vectors['IDN'])/3
                for name, vector in vectors.items():
                    route_vectors[route, method, radius, name] = vector
                    route_draws[route, method, radius, name] = weights[route] @ vector
            for name in ('I4', 'IDN', 'J', 'I6'):
                point = float(route_vectors['SNU', method, radius, name].mean() - route_vectors['Valley', method, radius, name].mean())
                sampled = route_draws['SNU', method, radius, name] - route_draws['Valley', method, radius, name]
                ci = interval(sampled)
                estimates.append({'method': method, 'radius_m': radius, 'quantity': name, 'point_pp': point,
                                  'ci95_low_pp': ci[0], 'ci95_high_pp': ci[1], 'saved_draws': 2000})
                for route in ('SNU', 'Valley'):
                    vector = route_vectors[route, method, radius, name]
                    for bi, block in enumerate(raw_blocks[route]):
                        keep = np.array([a not in block for a in anchors[route]])
                        other = 'Valley' if route == 'SNU' else 'SNU'
                        other_mean = route_vectors[other, method, radius, name].mean()
                        point_delete = vector[keep].mean() - other_mean if route == 'SNU' else other_mean - vector[keep].mean()
                        deletion.append({'method': method, 'radius_m': radius, 'quantity': name, 'deleted_route': route,
                                         'deleted_raw_block': bi, 'deleted_anchors': len(block), 'point_pp': float(point_delete)})
        for radius in (5, 15, 20, 25):
            sample_change = sum(sign*(route_draws[route, method, radius, 'J'] - route_draws[route, method, 10, 'J'])
                                for route, sign in (('SNU', 1), ('Valley', -1)))
            point_change = sum(sign*float((route_vectors[route, method, radius, 'J'] - route_vectors[route, method, 10, 'J']).mean())
                               for route, sign in (('SNU', 1), ('Valley', -1)))
            ci = interval(sample_change)
            direct.append({'method': method, 'comparison': f'J{radius}minus10', 'radius_m': radius,
                           'primary': int(radius == 25), 'point_pp': point_change, 'ci95_low_pp': ci[0], 'ci95_high_pp': ci[1]})
    lookup = {(r['method'], r['radius_m'], r['quantity']): r['point_pp'] for r in estimates}
    require(abs(lookup['C', 10, 'J'] - 54.78372898538481) <= 1e-11, 'old manuscript 10m contrast mismatch')
    for method in METHODS:
        for radius in RADII:
            require(abs(lookup[method, radius, 'I6'] - (2*lookup[method, radius, 'I4'] + lookup[method, radius, 'IDN'])/3) <= 1e-11, 'mixture identity mismatch')
    for route in ('SNU', 'Valley'):
        for condition in CONDITIONS:
            for anchor in anchors[route]:
                for method in METHODS:
                    vals = [result[route, condition, anchor, method, r] for r in RADII]
                    require(vals == sorted(vals), 'radius non-monotonicity')
    bands = []
    for route in ('SNU', 'Valley'):
        for group, conditions in [('morning_four', MORNING), ('day_night', DN), ('all_cross', MORNING+DN)]:
            for method in (*METHODS, 'RGB', 'Thermal'):
                selected = [r for r in diagnostic if r['route'] == route and r['condition'] in conditions and r['method'] == method]
                counts = Counter(r['band'] for r in selected)
                bands.append({'route': route, 'group': group, 'method': method, 'queries': len(selected),
                              'le10': counts['le10'], '10to25': counts['10to25'], 'gt25': counts['gt25'],
                              'R_at20_10m_percent': float(np.mean([r['r_at_20_10m'] for r in selected])*100)})
    deletion_index = {(r['method'], r['radius_m'], r['quantity'], r['deleted_route'], r['deleted_raw_block']): r for r in deletion}
    direct_deletion = []
    for row in deletion:
        if row['quantity'] == 'J' and row['radius_m'] == 25:
            reference = deletion_index[row['method'], 10, 'J', row['deleted_route'], row['deleted_raw_block']]
            direct_deletion.append({'method': row['method'], 'comparison': 'J25minus10',
                'deleted_route': row['deleted_route'], 'deleted_raw_block': row['deleted_raw_block'],
                'deleted_anchors': row['deleted_anchors'], 'point_pp': row['point_pp'] - reference['point_pp']})
    for name, output in [('query_radius', query_rows), ('direction_recall', direction_rows), ('contrasts', estimates),
                         ('direct_change', direct), ('block_deletions', deletion), ('block_members', block_members), ('bands', bands)]:
        csv_out(ROOT / f'analysis/{name}.csv', output)
    csv_out(ROOT / 'analysis/direct_block_deletions.csv', direct_deletion)
    json_out(ROOT / 'analysis/ANALYSIS_RECEIPT.json', {
        'passed_internal_checks': True, 'query_radius_records': len(query_rows), 'direction_cells': len(direction_rows),
        'contrast_records': len(estimates), 'direct_changes': len(direct),
        'anchors': {r: len(a) for r, a in anchors.items()}, 'raw_blocks': {r: len(b) for r, b in raw_blocks.items()},
        'saved_draws': 2000, 'protected_sources_unchanged': verify_sources(),
        'independent_audit': 'PENDING', 'no_new_predictions': True,
        'protocol_sha256': sha(ROOT / 'PROTOCOL.json'), 'script_sha256': sha(__file__),
    })
    print(json.dumps({'analysis_complete': True, 'main': [r for r in direct if r['primary'] and r['method'] in METHODS[:2]]}))

if __name__ == '__main__':
    analyze()
