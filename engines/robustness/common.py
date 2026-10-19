"""Portable cache-only I/O. No models, network, source mutations or downloads."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RADII=(5,10,15,20,25)
POLICIES=(("SALAD","full"),("Clique","full"),("SALAD","C20"))

def require(ok,message):
    if not ok: raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write_csv(path,rows):
    rows=list(rows);require(bool(rows),'empty table '+str(path));p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def load_inputs(root):
    root=Path(root)
    lock=read_json(root/'INPUTS_LOCK.json')
    for name,digest in lock['files'].items():require(sha(root/name)==digest,'input changed: '+name)
    s=dict(np.load(root/'inputs/scores.npz',allow_pickle=False))
    labels=dict(np.load(root/'inputs/geography.npz',allow_pickle=False))
    pred=dict(np.load(root/'inputs/predictions.npz',allow_pickle=False))
    queries=read_json(root/'inputs/queries.json')
    require(s['qids'].tolist()==pred['qids'].tolist()==[q['qid'] for q in queries],'query identity')
    return queries,s,labels['geo'],pred

def select(s,geo,qi,rep,policy):
    n=int(s['lengths'][qi]);ids=s['ids'][qi,:n];suffix='s' if rep=='SALAD' else 'c'
    dr,dt=s['dr_'+suffix][qi,:n],s['dt_'+suffix][qi,:n];g=geo[qi,:n]
    if policy=='C20':
        look={int(v):i for i,v in enumerate(ids)}
        ix=np.array([look[int(v)] for v in s['c20'][qi]],dtype=int)
        ids,dr,dt,g=ids[ix],dr[ix],dt[ix],g[ix]
    require(len(set(ids.tolist()))==len(ids) and np.isfinite(dr).all() and np.isfinite(dt).all() and np.isfinite(g).all(),'invalid candidate')
    return ids,dr,dt,g

def group(q):
    if q['route']=='NSAVP':return 'all_cross'
    a,b=q['condition'].split('_to_')
    return 'same' if a==b else 'morning_four' if 'M' in (a,b) else 'day_night'

def verify_protected(root=ROOT):
    root=Path(root);project=root.parents[1]
    records=read_json(root/'provenance/protected_before.json')
    for name,digest in records.items():require(sha(project/name)==digest,'protected source changed: '+name)
    return len(records)
