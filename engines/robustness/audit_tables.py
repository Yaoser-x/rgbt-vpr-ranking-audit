"""Independent prediction/aggregation replay from locked arrays and saved draws."""
from __future__ import annotations
import collections, time
from common import *

def main():
    start=time.monotonic();out=ROOT/'analysis';s=dict(np.load(ROOT/'inputs/scores.npz',allow_pickle=False));geo=np.load(ROOT/'inputs/geography.npz')['geo']
    pred=dict(np.load(ROOT/'inputs/predictions.npz',allow_pickle=False));queries=read_json(ROOT/'inputs/queries.json')
    qindex={q['qid']:i for i,q in enumerate(queries)};rows=read_csv(out/'query_results.csv');checked=0
    for r in rows:
        qi=qindex[r['qid']];n=int(s['lengths'][qi]);dist={int(g):float(d) for g,d in zip(s['ids'][qi,:n],geo[qi,:n])};radius=int(r['radius_m'])
        rep,policy=r['representation'],r['policy'];w=pred[f'{rep}_{policy}_weights'][:,qi];x=pred[f'{rep}_{policy}_max'][qi]
        def success(rank,k):return int(min(dist[int(g)] for g in rank[:k])<=radius)
        for prefix,rank in [('equal',w[5]),('max',x)]:
            for suffix,k in [('correct',1),('r5',5),('r20',20)]:require(int(r[prefix+'_'+suffix])==success(rank,k),'actual rank correctness');checked+=1
        for j in range(11):require(int(r[f'weight_{j:02d}_correct'])==success(w[j],1),'actual fixed weight correctness');checked+=1
    drawmeta=read_json(ROOT/'inputs/draws.json');grouped=collections.defaultdict(list)
    for r in rows:grouped[r['representation'],r['policy'],r['route'],r['radius_m']].append(r)
    tables=collections.defaultdict(list)
    for r in read_csv(out/'statistics.csv'):tables[r['representation'],r['policy'],r['route'],r['radius_m'],r['group']].append(r)
    for (rep,policy,route,radius,name),table in tables.items():
        pool=grouped[rep,policy,route,radius]
        if name=='all_records':selected=pool
        elif name=='all_cross':selected=[r for r in pool if r['condition'].split('_to_')[0]!=r['condition'].split('_to_')[1]]
        elif name=='same':selected=[r for r in pool if r['condition'].split('_to_')[0]==r['condition'].split('_to_')[1]]
        elif name=='morning_four':selected=[r for r in pool if r['condition'] in ('M_to_D','M_to_N','D_to_M','N_to_M')]
        elif name=='day_night':selected=[r for r in pool if r['condition'] in ('D_to_N','N_to_D')]
        else:selected=[r for r in pool if r['condition']==name]
        require(len(selected)==int(table[0]['queries']),'group selector')
        anchors=drawmeta[route]['anchors'];idx={a:i for i,a in enumerate(anchors)};w=np.load(ROOT/f'inputs/draws_{route}.npz')['weights']
        row_weights=w[:,[idx[r['anchor']] for r in selected]];draw_den=row_weights.sum(axis=1)
        ordinary=[r for r in table if r['metric']!='linear_impossible_fraction_of_C_error_C20']
        events=np.array([[float(q[r['metric']]) for r in ordinary] for q in selected]);values=(row_weights@events)/draw_den[:,None]*100
        limits=np.quantile(values,[.025,.975],axis=0)
        for j,r in enumerate(ordinary):
            require(int(r['count'])==int(events[:,j].sum()),'count aggregation');require(abs(float(r['value_percent'])-events[:,j].mean()*100)<=1e-11,'point aggregate')
            require(abs(float(r['ci95_low'])-limits[0,j])<=1e-11 and abs(float(r['ci95_high'])-limits[1,j])<=1e-11,'paired percentile interval');checked+=4
        ratio=next(r for r in table if r['metric']=='linear_impossible_fraction_of_C_error_C20')
        den=np.array([int(r['eligible_C_error_C20']) for r in selected]);num=np.array([int(r['eligible_linear_impossible']) for r in selected])
        sd,sn=row_weights@den,row_weights@num;valid=sd>0
        require(int(ratio['count'])==num.sum() and int(ratio['denominator'])==den.sum(),'conditional numerator/denominator')
        require(int(ratio['valid_draws'])==int(valid.sum()) and int(ratio['undefined_draws'])==int((~valid).sum()),'undefined denominator draws')
        if valid.any():
            lim=np.quantile(sn[valid]/sd[valid]*100,[.025,.975]);require(abs(float(ratio['ci95_low'])-lim[0])<=1e-11 and abs(float(ratio['ci95_high'])-lim[1])<=1e-11,'conditional interval')
        else:require(ratio['ci95_low']==ratio['ci95_high']=='','undefined interval must stay missing')
        checked+=5
    bounded=read_csv(out/'bounded_relative5m.csv');anchors=drawmeta['SNU']['anchors'];idx={a:i for i,a in enumerate(anchors)}
    w=np.load(ROOT/'inputs/draws_SNU.npz')['weights'][:,[idx[r['anchor']] for r in bounded]];den=w.sum(axis=1);valid=den>0
    for r in read_csv(out/'bounded_statistics.csv'):
        vals=np.array([int(x[r['metric']]) for x in bounded]);samples=w@vals
        lim=np.quantile(samples[valid]/den[valid]*100,[.025,.975])
        require(int(r['count'])==vals.sum() and abs(float(r['ci95_low'])-lim[0])<=1e-11 and abs(float(r['ci95_high'])-lim[1])<=1e-11,'bounded paired ratio');checked+=3
    write_json(out/'TABLE_AUDIT.json',{'passed':True,'query_radius_rows':len(rows),'checks':checked,'paired_summary_groups':len(tables),
          'source_sha256':sha(__file__),'source_arrays_locked':True,'elapsed_seconds':round(time.monotonic()-start,2),
          'method':'actual prediction identities independently joined to geographic distances; query-row weighted replay of saved spatial draws'})
    print({'tables_passed':True,'checks':checked,'seconds':round(time.monotonic()-start,1)})

if __name__=='__main__':main()
