"""Parallelize unchanged independent Fraction calls, preserving every condition."""
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import json
import os

import numpy as np
from common import read_json
from fraction_reference import reference, exact_winner


def initialize(root):
    global SCORES,GEO,QINDEX,GRID
    SCORES=dict(np.load(root/'inputs/scores.npz',allow_pickle=False))
    GEO=np.load(root/'inputs/geography.npz',allow_pickle=False)['geo']
    QINDEX={q['qid']:i for i,q in enumerate(read_json(root/'inputs/queries.json'))}
    GRID={}


def solve(item):
    rep,policy,qid,radius=[item[k] for k in ('representation','policy','qid','radius_m')]
    qi=QINDEX[qid];n=int(SCORES['lengths'][qi]);ids=SCORES['ids'][qi,:n]
    candidates=ids if policy=='full' else SCORES['c20'][qi]
    index={int(v):j for j,v in enumerate(ids)}
    ix=np.array([index[int(v)] for v in candidates]);suffix='s' if rep=='SALAD' else 'c'
    dr,dt=SCORES['dr_'+suffix][qi,ix],SCORES['dt_'+suffix][qi,ix]
    positive=GEO[qi,ix]<=radius
    stable=reference(dr,dt,positive);optimistic=reference(dr,dt,positive,True)
    key=rep,policy,qid
    if key not in GRID:GRID[key]=[exact_winner(dr,dt,Fraction(j,10)) for j in range(11)]
    return stable,optimistic,GRID[key]


def parallel_reference(root,out):
    workers=int(os.environ.get('PAPER_A_AUDIT_WORKERS','4'))
    with (out/'intervals.jsonl').open(encoding='utf-8') as stream:
        items=[json.loads(line) for line in stream]
    print('EXACT_REFERENCE_POOL',workers,'conditions',len(items),flush=True)
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(root,)) as pool:
        for item,result in zip(items,pool.map(solve,items,chunksize=20)):
            yield item,*result
