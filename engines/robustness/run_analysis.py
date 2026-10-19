"""Approved cache extension, with exact certificates and paired spatial summaries."""
from __future__ import annotations
import argparse, collections, json, time
from common import *
from exact_fusion import solve_query, grid_oracle


def analyze(root,out):
    start=time.monotonic();queries,s,geo,pred=load_inputs(root);out.mkdir(parents=True,exist_ok=True)
    identity={'inputs_lock_sha256':sha(root/'INPUTS_LOCK.json'),'runner_sha256':sha(__file__),
              'solver_sha256':sha(Path(__file__).with_name('exact_fusion.py'))}
    path=out/'RUN_IDENTITY.json'
    if path.exists():require(read_json(path)==identity,'resume identity conflict')
    else:write_json(path,identity)
    prior=read_csv(root/'inputs/old_10m_results.csv')
    old_map={(r['representation'],r['policy'],r['qid']):r for r in prior}
    fixed303={r['qid'] for r in prior if r['representation']=='SALAD' and r['policy']=='full' and
              r['route']=='SNU' and r['condition'] in ('M_to_D','M_to_N','D_to_M','N_to_M') and int(r['eligible_C_error_C20'])}
    require(len(fixed303)==303,'original fixed cohort')
    complete={}
    detail_path=out/'intervals.jsonl'
    if detail_path.exists():
        with detail_path.open(encoding='utf-8') as f:
            for line in f:
                item=json.loads(line);key=(item['representation'],item['policy'],item['qid'],item['radius_m'])
                require(key not in complete,'duplicate checkpoint row');complete[key]=item
    rows=[];numeric=[];replay_checks=0
    with detail_path.open('a',encoding='utf-8') as stream:
        for rep,policy in POLICIES:
            for qi,q in enumerate(queries):
                ids,dr,dt,g=select(s,geo,qi,rep,policy);ties=np.arange(len(ids));lookup={int(v):i for i,v in enumerate(ids)}
                w=pred[f'{rep}_{policy}_weights'][:,qi];x=pred[f'{rep}_{policy}_max'][qi]
                require(all(int(v) in lookup for v in w.ravel()) and all(int(v) in lookup for v in x),'rank escaped universe')
                wl=np.array([[lookup[int(v)] for v in a] for a in w]);xl=np.array([lookup[int(v)] for v in x])
                c20ix=[int(np.flatnonzero(s['ids'][qi,:int(s['lengths'][qi])]==v)[0]) for v in s['c20'][qi]]
                c20geo=geo[qi,c20ix]
                old=old_map[rep,policy,q['qid']]
                for radius in RADII:
                    p=g<=radius;key=(rep,policy,q['qid'],radius)
                    if key in complete:item=complete[key]
                    else:
                        item={'representation':rep,'policy':policy,'qid':q['qid'],'radius_m':radius,
                              'candidate_sha256':__import__('hashlib').sha256(ids.tobytes()).hexdigest(),
                              'solution':solve_query(dr,dt,p,ties),'grid':grid_oracle(dr,dt,p,ties)}
                        stream.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                    sol,grid=item['solution'],item['grid']
                    def good(rank,k):return int(np.any(g[rank[:k]]<=radius))
                    equal,maxi=good(wl[5],1),good(xl,1);avail=int(np.any(c20geo<=radius))
                    r={k:q[k] for k in ('qid','route','condition','anchor','block')}
                    r.update(radius_m=radius,representation=rep,policy=policy,gallery_size=len(ids),positive_count=int(p.sum()),
                             positive_available=int(p.any()),no_positive=int(not p.any()),equal_correct=equal,
                             equal_r5=good(wl[5],5),equal_r20=good(wl[5],20),max_correct=maxi,
                             max_r5=good(xl,5),max_r20=good(xl,20),equal_top1=int(w[5,0]),max_top1=int(x[0]),
                             equal_distance_m=float(g[wl[5,0]]),max_distance_m=float(g[xl[0]]),
                             rescue=int(not equal and maxi),harm=int(equal and not maxi),max_minus_equal=maxi-equal,
                             two_top1_oracle=int(good(wl[0],1) or good(wl[10],1)),grid11_oracle=int(grid['attainable']),
                             continuous_oracle=int(sol['attainable']),optimistic_oracle=int(sol['optimistic']),
                             linear_headroom=int(sol['attainable'])-equal,linear_impossible=int(p.any() and not sol['optimistic']),
                             optimistic_failure_including_no_positive=int(not sol['optimistic']),
                             tie_only_blocked=int(sol['optimistic'] and not sol['attainable']),salad_c20_available=avail,
                             eligible_C_error_C20=int(not equal and avail),eligible_linear_impossible=int(not equal and avail and not sol['optimistic']),
                             fixed303=int(q['qid'] in fixed303),max_rescue_beyond_linear=int(maxi and not sol['optimistic']),
                             every_positive_weakly_negative_dominated=int(sol['every_positive_weakly_negative_dominated']),
                             every_positive_strictly_both_negative_dominated=int(sol['every_positive_strictly_both_negative_dominated']),
                             undominated_positive_exists=int(sol['undominated_positive_exists']),classification=sol['classification'],
                             interval_shape=sol['shape'],interval_width=sol['width_float'])
                    for j in range(11):
                        actual=good(wl[j],1);exact=int(grid['correct'][j]);r[f'weight_{j:02d}_correct']=actual;r[f'exact_weight_{j:02d}_correct']=exact
                        if actual!=exact:numeric.append({'qid':q['qid'],'representation':rep,'policy':policy,'radius_m':radius,'weight_numerator':j,'actual':actual,'exact':exact})
                    require(r['continuous_oracle']>=r['grid11_oracle']>=equal,'oracle ordering')
                    if policy=='C20':require(r['equal_r20']==r['max_r20']==avail,'C20 R20')
                    if radius==10:
                        for k in set(r)&set(old):
                            if k in ('qid','route','condition','anchor','representation','policy','classification','interval_shape'):
                                require(str(r[k])==old[k],'10m text replay '+k)
                            else:require(float(r[k])==float(old[k]),'10m numeric replay '+k)
                            replay_checks+=1
                    rows.append(r)
                if qi%400==0:print(json.dumps({'rep':rep,'policy':policy,'queries':qi,'seconds':round(time.monotonic()-start,1)}),flush=True)
    require(len(rows)==34470,'incomplete conditions')
    write_csv(out/'query_results.csv',rows);write_json(out/'NUMERICAL_BOUNDARIES.json',numeric)
    bounded=[]
    bykey={(r['representation'],r['policy'],r['qid'],r['radius_m']):r for r in rows}
    for qi,q in enumerate(queries):
        if q['qid'] not in fixed303:continue
        ids,dr,dt,g=select(s,geo,qi,'SALAD','full')
        original=bykey['SALAD','full',q['qid'],10];a5=bykey['SALAD','full',q['qid'],5];a15=bykey['SALAD','full',q['qid'],15]
        fg=s['ids'][qi,:int(s['lengths'][qi])];lookup={int(v):i for i,v in enumerate(fg)}
        minimum=float(min(geo[qi,lookup[int(v)]] for v in s['c20'][qi]))
        winner_wrong=original['equal_distance_m']>15;candidate_guaranteed=minimum<=5;not_recoverable=not a15['optimistic_oracle']
        bounded.append({**{k:q[k] for k in ('qid','route','condition','anchor','block')},'original_C_distance_m':original['equal_distance_m'],
            'C20_nearest_distance_m':minimum,'guaranteed_C_wrong':int(winner_wrong),'guaranteed_C20_positive':int(candidate_guaranteed),
            'optimistic_15m_impossible':int(not_recoverable),'robust_candidate_present_failure':int(winner_wrong and candidate_guaranteed and not_recoverable),
            'oracle_guaranteed_attainable':a5['continuous_oracle'],'oracle_guaranteed_impossible':int(not_recoverable),
            'oracle_undetermined':int(not a5['continuous_oracle'] and not not_recoverable)})
    require(len(bounded)==303,'bounded fixed cohort');write_csv(out/'bounded_relative5m.csv',bounded)
    summarize(root,out,rows,bounded)
    write_json(out/'ANALYSIS_RECEIPT.json',{'status':'COMPLETE_PRODUCTION_PENDING_INDEPENDENT_AUDIT','rows':len(rows),
               'replayed_10m_cells':replay_checks,'numerical_boundary_cases':len(numeric),'elapsed_seconds':round(time.monotonic()-start,3),'identity':identity})
    print({'complete':True,'rows':len(rows),'seconds':round(time.monotonic()-start,2)},flush=True)


def summarize(root,out,rows,bounded):
    drawmeta=read_json(root/'inputs/draws.json');statistics=[];deletions=[];fixed=[];classes=[]
    metrics=('positive_available','no_positive','equal_correct','equal_r5','equal_r20','max_correct','max_r5','max_r20',
             'two_top1_oracle','grid11_oracle','continuous_oracle','optimistic_oracle','linear_headroom','linear_impossible',
             'optimistic_failure_including_no_positive','rescue','harm','max_minus_equal','eligible_C_error_C20','eligible_linear_impossible')
    grouped=collections.defaultdict(list)
    for r in rows:grouped[r['representation'],r['policy'],r['route'],r['radius_m']].append(r)
    for (rep,policy,route,radius),selected in grouped.items():
        anchors=drawmeta[route]['anchors'];blocks=drawmeta[route]['blocks'];w=np.load(root/f'inputs/draws_{route}.npz')['weights']
        groups={'all_records':selected}
        for name in ('same','morning_four','day_night','all_cross'):
            subset=[r for r in selected if group(r)==name or name=='all_cross' and group(r)!='same']
            if subset:groups[name]=subset
        for c in sorted({r['condition'] for r in selected}):groups[c]=[r for r in selected if r['condition']==c]
        for name,subset in groups.items():
            byanchor={a:[r for r in subset if str(r['anchor'])==a] for a in anchors};require(all(byanchor.values()),'anchor missing')
            def vector(metric):return np.array([np.mean([r[metric] for r in byanchor[a]]) for a in anchors])
            vectors=np.column_stack([vector(m) for m in metrics]);samples=w@vectors*100
            ci=np.quantile(samples,[.025,.975],axis=0)
            base={'representation':rep,'policy':policy,'route':route,'radius_m':radius,'group':name,'queries':len(subset)}
            for j,metric in enumerate(metrics):
                statistics.append({**base,'metric':metric,'count':sum(r[metric] for r in subset),'denominator':len(subset),
                     'value_percent':float(vectors[:,j].mean()*100),'ci95_low':float(ci[0,j]),'ci95_high':float(ci[1,j]),'valid_draws':2000,'undefined_draws':0})
                if metric in ('linear_headroom','linear_impossible','eligible_linear_impossible'):
                    for bi,block in enumerate(blocks):
                        keep=[a not in block for a in anchors]
                        deletions.append({**base,'metric':metric,'deleted_block':bi,'value_percent':float(vectors[keep,j].mean()*100)})
            den,num=vector('eligible_C_error_C20'),vector('eligible_linear_impossible');sd,sn=w@den,w@num;valid=sd>0
            limits=np.quantile(sn[valid]/sd[valid]*100,[.025,.975]) if valid.any() else [None,None]
            statistics.append({**base,'metric':'linear_impossible_fraction_of_C_error_C20','count':sum(r['eligible_linear_impossible'] for r in subset),
              'denominator':sum(r['eligible_C_error_C20'] for r in subset),'value_percent':float(num.mean()/den.mean()*100) if den.mean()>0 else None,
              'ci95_low':None if limits[0] is None else float(limits[0]),'ci95_high':None if limits[1] is None else float(limits[1]),
              'valid_draws':int(valid.sum()),'undefined_draws':int((~valid).sum())})
            for cls in sorted({r['classification'] for r in subset}):classes.append({**base,'classification':cls,'count':sum(r['classification']==cls for r in subset)})
            for j in range(11):fixed.append({**base,'alpha_numerator':j,'alpha_denominator':10,'correct':sum(r[f'weight_{j:02d}_correct'] for r in subset),
                'R1_percent':float(vector(f'weight_{j:02d}_correct').mean()*100)})
    write_csv(out/'statistics.csv',statistics);write_csv(out/'block_deletions.csv',deletions);write_csv(out/'fixed_weights.csv',fixed);write_csv(out/'classifications.csv',classes)
    transitions=[]
    for rep,policy in POLICIES:
        for radius in RADII:
            selected=[r for r in rows if r['representation']==rep and r['policy']==policy and r['radius_m']==radius and r['fixed303']]
            transitions.append({'representation':rep,'policy':policy,'radius_m':radius,'fixed_10m_cohort':len(selected),
                **{m:sum(r[m] for r in selected) for m in ('positive_available','equal_correct','continuous_oracle','optimistic_oracle','linear_impossible','eligible_C_error_C20','eligible_linear_impossible')}})
    write_csv(out/'fixed303_transitions.csv',transitions)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evidence-root',type=Path,default=ROOT);parser.add_argument('--output-root',type=Path)
    args=parser.parse_args();root=args.evidence_root.resolve();out=(args.output_root or root/'analysis').resolve()
    require(out!=root/'inputs' and not (root/'MANIFEST.json').exists(),'sealed/input output forbidden')
    analyze(root,out)
