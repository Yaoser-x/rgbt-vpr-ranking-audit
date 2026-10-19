"""Independent Fraction audit of every radius; no production solver imported."""
from __future__ import annotations
import argparse, collections, json, time
from fractions import Fraction
from common import ROOT, RADII, sha, read_json, read_csv, write_json, require, np, Path, verify_protected
from fraction_reference import reference, parse_production_intervals, exact_winner, self_test


def audit(root,out,portable=False):
    start=time.monotonic();checks=0
    audited_sources=('audit.py','common.py','fraction_reference.py','exact_fusion.py','run_analysis.py')
    source_snapshot={name:sha(Path(__file__).parent/name) for name in audited_sources}
    lock=read_json(root/'INPUTS_LOCK.json')
    for name,digest in lock['files'].items():require(sha(root/name)==digest,'input lock mismatch')
    scores=dict(np.load(root/'inputs/scores.npz',allow_pickle=False));geo=np.load(root/'inputs/geography.npz')['geo']
    queries=read_json(root/'inputs/queries.json');index={q['qid']:i for i,q in enumerate(queries)}
    rows=read_csv(out/'query_results.csv');keycols=('representation','policy','qid','radius_m')
    bykey={(r['representation'],r['policy'],r['qid'],int(r['radius_m'])):r for r in rows}
    require(len(bykey)==len(rows)==34470,'row identity or incomplete conditions')
    # Reference APIs are frozen before the audit begins; no mid-run source edits.
    tests=self_test();count=0;grid_cache={};seen=set();last=None
    from reference_pool import parallel_reference
    for item, stable, optimistic, winners in parallel_reference(root,out):
        rep,policy,qid,radius=[item[k] for k in keycols];key=(rep,policy,qid,radius)
        require(key not in seen and key in bykey,'certificate identity');seen.add(key)
        qi=index[qid];n=int(scores['lengths'][qi]);gids=scores['ids'][qi,:n]
        candidates=gids if policy=='full' else scores['c20'][qi]
        location={int(v):i for i,v in enumerate(gids)};ix=np.array([location[int(v)] for v in candidates])
        suffix='s' if rep=='SALAD' else 'c';dr=scores['dr_'+suffix][qi,ix];dt=scores['dt_'+suffix][qi,ix]
        distances=geo[qi,ix];positive=distances<=radius
        require(item['candidate_sha256']==__import__('hashlib').sha256(candidates.tobytes()).hexdigest(),'candidate identity')
        sol=item['solution']
        require(stable['intervals']==parse_production_intervals(sol['attainable_intervals']),'stable interval mismatch '+repr(key))
        require(optimistic['intervals']==parse_production_intervals(sol['optimistic_intervals']),'optimistic interval mismatch '+repr(key))
        r=bykey[key];require(int(r['continuous_oracle'])==bool(stable['intervals']) and int(r['optimistic_oracle'])==bool(optimistic['intervals']),'oracle bool mismatch')
        gkey=(rep,policy,qid)
        grid_cache[gkey]=winners
        require(winners==item['grid']['winners'],'exact grid winners')
        require([bool(positive[v]) for v in winners]==item['grid']['correct'],'grid correctness')
        require(int(r['positive_count'])==int(positive.sum()),'positive count')
        require(int(r['max_correct'])-int(r['equal_correct'])==int(r['rescue'])-int(r['harm']),'rescue harm identity')
        if policy=='C20':require(int(r['equal_r20'])==int(r['max_r20'])==int(positive.any()),'C20 availability')
        for j in range(11):require(int(r[f'exact_weight_{j:02d}_correct'])==int(positive[winners[j]]),'stored exact fixed score')
        checks+=20;count+=1
        if count%2000==0:print({'fraction_audited':count,'seconds':round(time.monotonic()-start,1)},flush=True)
    require(len(seen)==34470,'certificate coverage')
    for rep,policy,qid in grid_cache:
        selected=[bykey[rep,policy,qid,r] for r in RADII]
        for metric in ('positive_count','positive_available','equal_correct','equal_r5','equal_r20','max_correct','max_r5','max_r20','continuous_oracle','optimistic_oracle'):
            vals=[float(r[metric]) for r in selected];require(vals==sorted(vals),'nonmonotone '+metric+' '+qid);checks+=1
    old=read_csv(root/'inputs/old_10m_results.csv');replay=0
    for r in old:
        new=bykey[r['representation'],r['policy'],r['qid'],10]
        for k,v in new.items():
            if k not in r:continue
            if k in ('qid','route','condition','anchor','representation','policy','classification','interval_shape'):require(v==r[k],'10m string replay')
            else:require(float(v)==float(r[k]),'10m numeric replay')
            replay+=1
    fixed_ids={r['qid'] for r in rows if r['representation']=='SALAD' and r['policy']=='full' and r['radius_m']=='10' and r['fixed303']=='1'}
    require(len(fixed_ids)==303,'fixed 303 identity')
    bounded=read_csv(out/'bounded_relative5m.csv');require({r['qid'] for r in bounded}==fixed_ids,'bounded cohort identity')
    for b in bounded:
        qi=index[b['qid']];n=int(scores['lengths'][qi]);ids=scores['ids'][qi,:n];loc={int(v):i for i,v in enumerate(ids)}
        minc=min(float(geo[qi,loc[int(v)]]) for v in scores['c20'][qi]);r15=bykey['SALAD','full',b['qid'],15];r5=bykey['SALAD','full',b['qid'],5]
        expected=int(float(b['original_C_distance_m'])>15 and minc<=5 and not int(r15['optimistic_oracle']))
        require(expected==int(b['robust_candidate_present_failure']),'bounded predicate')
        require(not (int(r5['continuous_oracle']) and not int(r15['optimistic_oracle'])),'bounded contradiction');checks+=2
    # Translation/rotation is an invariance check, not a model of measured error.
    pos=np.array([[0.,0.],[3.,4.],[15.,0.]]);offset=np.array([123.,-47.]);rot=np.array([[0.,-1.],[1.,0.]])
    require(np.allclose(np.linalg.norm(pos[1:]-pos[0],axis=1),np.linalg.norm((pos@rot+offset)[1:]-(pos@rot+offset)[0],axis=1),rtol=0,atol=1e-12),'rigid transform invariant')
    for d in np.array([0.,5.,10.,15.,20.]):require((d<=5)<= (d<=10)<= (d<=15),'nested label test')
    require(source_snapshot=={name:sha(Path(__file__).parent/name) for name in audited_sources},'audited source changed during audit')
    write_json(out/'INDEPENDENT_AUDIT.json',{'passed':True,'conditions':count,'fraction_interval_unions_checked':count*2,
       'exact_grid_winner_sets':len(grid_cache),'independent_checks':checks,'10m_replayed_cells':replay,'synthetic_reference_tests':tests,
       'bounded_fixed_records':303,'source_snapshot':source_snapshot,'elapsed_seconds':round(time.monotonic()-start,3),
       'basis':'independent Fraction implementation; assistant computation, not external human replication',
       'inputs_lock_sha256':sha(root/'INPUTS_LOCK.json'),'query_results_sha256':sha(out/'query_results.csv'),
       'certificates_sha256':sha(out/'intervals.jsonl'),'protected_sources_unchanged':
           'portable run: input locks checked; original private source tree unavailable' if portable else verify_protected(root)})
    print({'audit_passed':True,'conditions':count,'seconds':round(time.monotonic()-start,1)},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence-root',type=Path,default=ROOT);p.add_argument('--output-root',type=Path);p.add_argument('--portable',action='store_true')
    a=p.parse_args();root=a.evidence_root.resolve();audit(root,(a.output_root or root/'analysis').resolve(),a.portable)
