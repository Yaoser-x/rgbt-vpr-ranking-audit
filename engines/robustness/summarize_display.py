"""Rebuild existing radius and bounded display summaries without plotting or case selection."""
from common import *
POPS=(('SNU','morning_four'),('SNU','day_night'),('Valley','all_cross'),('NSAVP','all_cross'))

def main(root,out):
    stats=read_csv(out/'statistics.csv');rows=read_csv(out/'query_results.csv');bounded=read_csv(out/'bounded_relative5m.csv')
    def stat(rep,route,group,radius,metric):
        values=[r for r in stats if (r['representation'],r['policy'],r['route'],r['group'],int(r['radius_m']),r['metric'])==(rep,'full',route,group,radius,metric)]
        require(len(values)==1,'nonunique summary selector');return values[0]
    summaries=[]
    for rep in ('SALAD','Clique'):
        for route,grp in POPS:
            for radius in RADII:
                sel=lambda m:stat(rep,route,grp,radius,m)
                ratio=sel('linear_impossible_fraction_of_C_error_C20')
                summaries.append({'representation':rep,'route':route,'group':grp,'radius_m':radius,'queries':sel('equal_correct')['queries'],
                    **{m:sel(m)['count'] for m in ('positive_available','no_positive','equal_correct','continuous_oracle','optimistic_oracle','linear_impossible','linear_headroom','max_correct')},
                    'eligible_C_error_C20':ratio['denominator'],'eligible_linear_impossible':ratio['count'],'conditional_percent':ratio['value_percent'],
                    'conditional_ci95_low':ratio['ci95_low'],'conditional_ci95_high':ratio['ci95_high'],
                    'undefined_ratio_draws':ratio['undefined_draws']})
    write_csv(out/'main_radius_summary.csv',summaries)
    # Paired ratio over the fixed303 subset: original anchor draws, not point-wise resampling.
    meta=read_json(root/'inputs/draws.json')['SNU'];anchors=meta['anchors'];weights=np.load(root/'inputs/draws_SNU.npz')['weights']
    metrics=['guaranteed_C_wrong','guaranteed_C20_positive','optimistic_15m_impossible','robust_candidate_present_failure',
             'oracle_guaranteed_attainable','oracle_guaranteed_impossible','oracle_undetermined']
    bs=[];den=np.array([sum(r['anchor']==a for r in bounded)/4 for a in anchors]);sample_den=weights@den
    for metric in metrics:
        num=np.array([sum(int(r[metric]) for r in bounded if r['anchor']==a)/4 for a in anchors]);sample_num=weights@num;valid=sample_den>0
        limits=np.quantile(sample_num[valid]/sample_den[valid]*100,[.025,.975]);count=sum(int(r[metric]) for r in bounded)
        bs.append({'metric':metric,'count':count,'denominator':303,'value_percent':100*count/303,'ci95_low':float(limits[0]),'ci95_high':float(limits[1]),
                   'valid_draws':int(valid.sum()),'undefined_draws':int((~valid).sum())})
    write_csv(out/'bounded_statistics.csv',bs)
    spatial=[]
    for bi,block in enumerate(meta['blocks']):
        subset=[r for r in bounded if r['anchor'] in block]
        spatial.append({'raw_block':bi,'fixed303_records':len(subset),'distinct_anchors':len({r['anchor'] for r in subset}),
             'robust_failures':sum(int(r['robust_candidate_present_failure']) for r in subset)})
    write_csv(out/'bounded_spatial_counts.csv',spatial)

if __name__ == '__main__':
    main(ROOT, ROOT/'analysis')
