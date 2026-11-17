"""Offline reconstruction from frozen distances and metadata, never model inference."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
REPRESENTATIONS = {'SALAD': ['dr_s', 'dt_s'], 'Clique': ['dr_c', 'dt_c']}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def validate(root):
    manifest = read(root/'INPUT_MANIFEST.json')
    require(manifest.get('schema') == 1, 'Unsupported input schema')
    require(manifest.get('representations') == REPRESENTATIONS, 'Wrong representation mapping')
    for name, record in manifest['files'].items():
        member = (root/name).resolve()
        require(member.is_relative_to(root), 'Input member escapes package')
        require(member.is_file(), 'Missing input member: '+name)
        require(member.stat().st_size == record['bytes'] and sha(member) == record['sha256'], 'Input SHA mismatch: '+name)
    for engine in ('robustness', 'capability'):
        folder = root/'payload'/engine
        scores = np.load(folder/'inputs/scores.npz', allow_pickle=False)
        queries = read(folder/'inputs/queries.json')
        require(scores['qids'].tolist() == [q['qid'] for q in queries], 'Query identity mismatch')
        require(set(REPRESENTATIONS) == {'SALAD', 'Clique'}, 'Representation contract')
        for rkey, tkey in REPRESENTATIONS.values():
            require(scores[rkey].dtype == np.float64 and scores[tkey].dtype == np.float64, 'Distance dtype must be float64')
            require(scores[rkey].shape == scores[tkey].shape == scores['ids'].shape, 'Distance/ID shape mismatch')
        require(len(set(scores['qids'].tolist())) == len(queries) == 2298, 'Frozen query cohort mismatch')
    return len(manifest['files'])


def new_output(root, out):
    require(not out.exists(), 'Output must not already exist')
    require(not out.is_relative_to(root) and not root.is_relative_to(out), 'Input and output must be disjoint')
    for parent in (out, *out.parents):
        require(not (parent/'MANIFEST.json').exists(), 'Output is beneath sealed evidence')
    out.mkdir(parents=True)


def compare_value(actual, expected, label):
    if actual == expected:
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        require(actual.keys() == expected.keys(), 'Key mismatch: '+label)
        for key in actual:
            compare_value(actual[key], expected[key], label+'/'+str(key))
    elif isinstance(actual, list) and isinstance(expected, list):
        require(len(actual) == len(expected), 'Length mismatch: '+label)
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare_value(a, b, label+'/'+str(i))
    else:
        # Identity strings and integer fields are exact; tolerance applies only to finite floating summaries.
        if re.fullmatch(r'[+-]?\d+', str(actual)) and re.fullmatch(r'[+-]?\d+', str(expected)):
            require(int(actual) == int(expected), 'Integer mismatch: '+label)
            return
        try:
            a, b = float(actual), float(expected)
        except (TypeError, ValueError):
            raise RuntimeError('Value mismatch: '+label+': '+repr(actual)+' != '+repr(expected))
        require(math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-11, 'Float mismatch: '+label)


def compare_csv(actual, expected):
    with actual.open(encoding='utf-8-sig', newline='') as a, expected.open(encoding='utf-8-sig', newline='') as b:
        ar, br = csv.reader(a), csv.reader(b)
        rows, cells = 0, 0
        from itertools import zip_longest
        for i, (x, y) in enumerate(zip_longest(ar, br)):
            require(x is not None and y is not None and len(x) == len(y), 'CSV shape mismatch: '+str(expected))
            for j, (v, w) in enumerate(zip(x, y)):
                compare_value(v, w, expected.name+f'/{i}/{j}')
                cells += 1
            rows += 1
    return dict(file=expected.name, rows=rows-1, cells=cells, byte_identical=sha(actual)==sha(expected))


def run_stage(out, name, args):
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2',
               PYTHONPATH=str(ROOT/'guard'), PAPER_A_READ_ROOTS=json.dumps([str(out), str(ROOT)]),
               PAPER_A_ACCESS_LOG=str(out/'access'/f'{name}.json'))
    print('START', name, flush=True)
    log = out/'logs'/f'{name}.log'; log.parent.mkdir(exist_ok=True)
    start = time.monotonic()
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', *map(str, args)], cwd=out,
                                env=env, stdout=stream, stderr=subprocess.STDOUT)
    require(result.returncode == 0, 'Stage failed: '+name+'; see '+str(log))
    print('DONE', name, round(time.monotonic()-start, 2), 's', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--audit-workers',type=int,default=4)
    args = parser.parse_args()
    root, out = args.input_root.resolve(), args.output_root.resolve()
    require(1 <= args.audit_workers <= (os.cpu_count() or 1),'Invalid audit worker count')
    os.environ['PAPER_A_AUDIT_WORKERS']=str(args.audit_workers)
    require(root == ROOT, 'Run the entrypoint shipped with this input root')
    locked = validate(root)
    new_output(root, out)
    write(out/'RUN.json', dict(status='RUNNING', input_manifest_sha256=sha(root/'INPUT_MANIFEST.json'),
          locked_members=locked, inference=False, training=False, downloads=False, new_draws=False,
          representation_mapping=REPRESENTATIONS, python=sys.version, numpy=np.__version__))
    for engine in ('robustness', 'capability', 'reconstruction'):
        shutil.copytree(root/'payload'/engine, out/engine)
        shutil.copytree(root/'engines'/engine, out/engine/'code', dirs_exist_ok=True)
    cap, robust = out/'capability', out/'robustness'
    members = {p.relative_to(cap).as_posix(): sha(p) for folder in ('inputs', 'code', 'training')
               for p in (cap/folder).rglob('*') if p.is_file()}
    write(cap/'PORTABLE_MEMBERS.json', members)
    reconstruction = out/'reconstruction'
    write(reconstruction/'PORTABLE_MEMBERS.json', {
        p.relative_to(reconstruction).as_posix(): sha(p) for p in reconstruction.rglob('*') if p.is_file()})
    run_stage(out, 'acquisition-contrast-replay', [reconstruction/'code/reconstruct.py'])
    run_stage(out, 'robustness-produce', [robust/'code/run_analysis.py', '--evidence-root', robust, '--output-root', robust/'analysis'])
    run_stage(out, 'robustness-display-summaries', [robust/'code/summarize_display.py'])
    # New production output must match the scalar rows consumed by the monotone stage.
    compare_csv(robust/'analysis/query_results.csv', cap/'inputs/legacy_results.csv')
    for name, command in [('capability-produce', ['analyze.py', 'ability']),
                          ('official-overlap', ['analyze.py', 'overlap']),
                          ('sampling-replay', ['sensitivity.py']),
                          ('training-prediction-aggregation', ['evaluate_training.py'])]:
        run_stage(out, name, [cap/'code'/command[0], *command[1:]])
    comparisons = []
    source_only=[]
    for group, actual in [('robustness', robust/'analysis'), ('capability', cap/'analysis'),
                          ('training', cap/'training'), ('reconstruction', reconstruction/'analysis')]:
        for expected in sorted((root/'expected'/group).glob('*.csv')):
            if (actual/expected.name).exists():
                comparisons.append(dict(group=group, **compare_csv(actual/expected.name, expected)))
            else:
                source_only.append(dict(group=group,file=expected.name,
                    status='HISTORICAL_SOURCE_ONLY_NOT_A_RECALCULATION'))
    compare_value(read(cap/'training/DECISION.json'), read(root/'expected/training/DECISION.json'), 'training-decision')
    # Keep documentary execution histories separate from recomputed predictions.
    from display_audit import build_display_report
    display = build_display_report(root, out)
    with (cap/'analysis/query_results.csv').open(encoding='utf-8',newline='') as stream:
        morning=[r for r in csv.DictReader(stream) if r['route']=='SNU' and r['condition'] in ('M_to_D','D_to_M','M_to_N','N_to_M')
                 and r['policy']=='full' and r['representation']=='SALAD' and r['radius_m']=='10']
    with (robust/'analysis/query_results.csv').open(encoding='utf-8',newline='') as stream:
        coverage=sum(int(r['equal_r20']) for r in csv.DictReader(stream) if r['route']=='SNU' and r['condition'] in ('M_to_D','D_to_M','M_to_N','N_to_M')
                     and r['policy']=='full' and r['representation']=='SALAD' and r['radius_m']=='10')
    core=dict(records=len(morning),anchors=len({r['anchor'] for r in morning}),
              equal_correct=sum(int(r['equal_correct']) for r in morning),top20_coverage=coverage,
              scalar=sum(int(r['scalar_stable']) for r in morning),monotone_D=sum(int(r['D_stable']) for r in morning),
              strict_both_obstruction=sum(int(r['D_impossible']) for r in morning))
    require(core==dict(records=532,anchors=133,equal_correct=216,top20_coverage=519,scalar=280,monotone_D=295,strict_both_obstruction=237),'Core result mismatch')
    write(out/'CORE_RESULTS.json',core)
    run_stage(out, 'robustness-fraction-audit', [robust/'code/audit.py', '--evidence-root', robust, '--output-root', robust/'analysis', '--portable'])
    run_stage(out, 'robustness-table-audit', [robust/'code/audit_tables.py'])
    run_stage(out, 'capability-independent-audit', [cap/'code/audit.py'])
    validate(root)
    write(out/'COMPARISONS.json', comparisons)
    write(out/'SOURCE_ONLY_REFERENCES.json',source_only)
    write(out/'COMPLETION.json', dict(status='PASSED_FROZEN_DISTANCE_RECONSTRUCTION',
          comparison_files=len(comparisons), comparison_cells=sum(x['cells'] for x in comparisons),
          display_items=display, independent_external_replication=False, raw_image_to_result_reproduction=False,
          public_release=False, input_manifest_sha256=sha(root/'INPUT_MANIFEST.json')))
    print('RECONSTRUCTION PASSED', out, flush=True)


if __name__ == '__main__':
    main()
