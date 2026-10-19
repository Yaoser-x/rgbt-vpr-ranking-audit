"""E: one-axis deterministic thinning and alternative spatial-block lengths."""
from __future__ import annotations
import time
from collections import defaultdict
from io_utils import *
from analyze import record, METRICS

def greedy_positions(ids, frames, spacing):
    kept, positions = [], []
    for i in ids:
        f = frames[int(i)]
        point = np.array([float(f['row']['x_m']), float(f['row']['y_m'])])
        if all(np.linalg.norm(point - other) > spacing for other in positions):
            kept.append(int(i)); positions.append(point)
    return kept

def main():
    start = time.monotonic()
    queries, scores = load_scores(); geo = load_labels()
    meta = read_json(ROOT / 'inputs/frame_metadata.json'); frames = meta['frames']
    qbank = {q['qid']: q['query_index'] for q in meta['queries']}
    prior = read_csv(ROOT / 'analysis/query_results.csv')
    baseline = [r for r in prior if r['policy'] == 'full' and r['route'] != 'NSAVP']
    numeric = set(METRICS) | {'positive_count', 'equal_top1', 'fixed303', 'original_failure', 'candidate_present_failure',
                              'radius_m', 'gallery_size', 'block'}
    for r in baseline:
        for k in numeric:
            r[k] = int(r[k])
        r['equal_distance_m'] = float(r['equal_distance_m'])
    fixed303 = {r['qid'] for r in baseline if r['fixed303']}
    rows, selections = [], []
    for route in ('SNU', 'Valley'):
        anchors = [q for q in queries if q['route'] == route and q['condition'] == 'M_to_M']
        for spacing in (20, 40):
            bank_ids = [qbank[q['qid']] for q in anchors]
            kept = set(greedy_positions(bank_ids, frames, spacing))
            keep_anchor = {q['anchor'] for q in anchors if qbank[q['qid']] in kept}
            for q in anchors:
                selections.append(dict(axis='query', route=route, sequence='Morning', spacing_m=spacing,
                    identity=q['anchor'], retained=int(q['anchor'] in keep_anchor)))
            for old in baseline:
                if old['route'] == route and old['anchor'] in keep_anchor:
                    rows.append(dict(old, setting=f'query_{spacing}m'))
    gallery_cache = {}
    out = ROOT / 'analysis'
    with (out / 'sensitivity_certificates.jsonl').open('w', encoding='utf-8') as stream:
        for qi, q in enumerate(queries):
            if q['route'] == 'NSAVP':
                continue
            full_ids = scores['ids'][qi, :int(scores['lengths'][qi])]
            for spacing in (10, 20):
                token = (q['route'], frames[int(full_ids[0])]['sequence'], spacing)
                if token not in gallery_cache:
                    kept = set(greedy_positions(full_ids, frames, spacing))
                    gallery_cache[token] = np.array([i for i, b in enumerate(full_ids) if int(b) in kept])
                    for bank in full_ids:
                        selections.append(dict(axis='gallery', route=q['route'], sequence=token[1], spacing_m=spacing,
                            identity=int(bank), retained=int(bank) in kept))
                ix = gallery_cache[token]
                for rep in ('SALAD', 'Clique'):
                    ids, r, t, ix = select(scores, qi, rep, ix)
                    for radius in RADII:
                        row, cert = record(q, qi, rep, 'full', ids, r, t, ix, geo, ix, radius, None, fixed303)
                        row['setting'] = f'gallery_{spacing}m'; rows.append(row)
                        cert['setting'] = row['setting']
                        cert['candidate_indices'] = ix.tolist()
                        stream.write(json.dumps(cert, allow_nan=False) + '\n')
            if qi % 300 == 0:
                print('SENSITIVITY', qi, 'seconds', round(time.monotonic() - start, 1), flush=True)
    write_csv(out / 'sensitivity_queries.csv', rows)
    write_csv(out / 'sampling_selections.csv', selections)
    write_csv(out / 'sensitivity.csv', summarize(rows, METRICS, by=('setting', 'representation', 'radius_m')))
    # The raw pre-cohort anchor ID, not compact retained-row index, defines blocks.
    block_stats, definitions = [], []
    for route in ('SNU', 'Valley'):
        for length in (5, 20):
            raw_ids = read_json(ROOT / f'inputs/raw_anchors_{route}.json')
            require(len(set(raw_ids)) == len(raw_ids), 'raw anchor duplicate')
            blocks = [raw_ids[i:i + length] for i in range(0, len(raw_ids), length)]
            picks = np.load(ROOT / f'inputs/block_draws_{route}_{length}.npz', allow_pickle=False)['picks']
            require(picks.shape == (2000, len(blocks)), 'frozen alternative draw shape')
            np.savez_compressed(out / f'block_draws_{route}_{length}.npz', picks=picks)
            for bi, block in enumerate(blocks):
                definitions.append(dict(route=route, raw_block_length=length, block=bi, anchors='|'.join(block)))
            buckets = defaultdict(list)
            for row in baseline:
                if row['route'] == route:
                    buckets[(row['representation'], row['radius_m'], group(row))].append(row)
            for (rep, radius, name), subset in buckets.items():
                for metric in METRICS:
                    numerators = np.array([sum(int(r[metric]) for r in subset if str(r['anchor']) in b) for b in blocks], dtype=float)
                    denominators = np.array([sum(str(r['anchor']) in b for r in subset) for b in blocks], dtype=float)
                    require(denominators.sum() == len(subset), 'raw block membership loss')
                    sn, sd = numerators[picks].sum(1), denominators[picks].sum(1); valid = sd > 0
                    bounds = np.quantile(sn[valid] / sd[valid] * 100, [.025, .975]) if valid.any() else [None, None]
                    block_stats.append(dict(route=route, representation=rep, radius_m=radius, group=name,
                        raw_block_length=length, blocks=len(blocks), metric=metric, count=int(numerators.sum()),
                        denominator=int(denominators.sum()), value_percent=float(numerators.sum() / denominators.sum() * 100),
                        ci95_low=None if bounds[0] is None else float(bounds[0]),
                        ci95_high=None if bounds[1] is None else float(bounds[1]), undefined_draws=int((~valid).sum())))
    write_csv(out / 'block_sensitivity.csv', block_stats)
    write_csv(out / 'block_definitions.csv', definitions)
    write_json(out / 'SENSITIVITY_RECEIPT.json', dict(rows=len(rows), seconds=time.monotonic() - start,
        settings=['query_20m', 'query_40m', 'gallery_10m', 'gallery_20m'],
        denser_sampling='NOT_TESTED', NSAVP='not included in STheReO XY thinning', independent_audit='PENDING'))
    print('SENSITIVITY_COMPLETE', len(rows), flush=True)

if __name__ == '__main__':
    main()
