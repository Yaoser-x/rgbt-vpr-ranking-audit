"""Bind displayed cells to recomputed rows, with explicit documentary exceptions."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

import numpy as np


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def tables(text):
    pattern = r'\*\*Table\s+(S?\d+)\.\s*(.*?)\*\*[^\n]*\n\s*\n((?:\|[^\n]*\|[ \t]*(?:\n|$))+)'
    return {m[1]: [[c.strip() for c in line.strip().strip('|').split('|')]
                  for line in m[3].strip().splitlines()][2:] for m in re.finditer(pattern, text)}


def group(r):
    if r['route'] == 'NSAVP':
        return 'all_cross'
    a, b = r['condition'].split('_to_')
    return 'same' if a == b else 'morning_four' if 'M' in (a, b) else 'day_night'


def build_display_report(root, out):
    from reproduce import compare_value, require, read, write
    main = tables((root/'reference/manuscript.md').read_text(encoding='utf-8'))
    si = tables((root/'reference/supplementary.md').read_text(encoding='utf-8'))
    require(len(main)==2 and len(si)==33, 'Displayed table inventory')
    rr = rows(out/'robustness/analysis/query_results.csv')
    stats = rows(out/'robustness/analysis/statistics.csv')
    cc = rows(out/'capability/analysis/statistics.csv')
    summary = rows(out/'capability/analysis/sensitivity.csv')
    overlaps = rows(out/'capability/analysis/model_overlap.csv')
    seeds = rows(out/'capability/training/seed_summary.csv')
    draws = read(out/'capability/inputs/draws.json')
    checks = Counter()

    def cell(item, actual, expected):
        if isinstance(actual, (float,np.floating)) and isinstance(expected,str):
            # A printed cell is compared at its stated decimal precision.
            normalized = expected.replace('−','-')
            if re.fullmatch(r'-?\d+\.\d+', normalized):
                actual = f'{actual:.{len(normalized.split(".")[1])}f}'
                expected = normalized
        compare_value(actual, expected, item)
        checks[item] += 1

    def select(rep, population, policy='full', radius=10):
        route, name = population.split(' ',1)
        return [r for r in rr if r['representation']==rep and r['policy']==policy and int(r['radius_m'])==int(radius)
                and r['route']==route and (r['condition']==name or group(r)==name or name=='all_cross' and group(r)!='same')]

    def count(data, key):
        return sum(int(r[key]) for r in data)

    def stat(data, rep, population, metric, radius=10, policy='full', **extra):
        route, name=population.split(' ',1)
        matches=[r for r in data if r['representation']==rep and r['route']==route and r['group']==name
                 and int(r['radius_m'])==int(radius) and r['metric']==metric
                 and ('policy' not in r or r['policy']==policy) and all(r[k]==str(v) for k,v in extra.items())]
        require(len(matches)==1, 'Ambiguous display selector '+repr((rep,population,metric,radius,extra)))
        return matches[0]

    def interval(r):
        return f"{float(r['value_percent']):.2f} [{float(r['ci95_low']):.2f}, {float(r['ci95_high']):.2f}]"

    for item in ('S1','S3'):
        fields = ('equal_correct','two_top1_oracle','grid11_oracle','continuous_oracle','linear_impossible','max_correct') if item=='S1' else (
            'equal_correct','continuous_oracle','linear_impossible','every_positive_weakly_negative_dominated','undominated_positive_exists','max_correct')
        for r in si[item]:
            data=select(r[0],r[1]); cell(item,len(data),r[2])
            for key,expected in zip(fields,r[3:]):
                source=[x for x in data if int(x['linear_impossible'])] if key in ('every_positive_weakly_negative_dominated','undominated_positive_exists') else data
                cell(item,count(source,key),expected)
    for r in si['S2']:
        data=select(r[0],r[1]); st=stat(stats,r[0],r[1],'linear_impossible_fraction_of_C_error_C20')
        impossible=[x for x in data if int(x['linear_impossible'])]
        cell('S2',count(impossible,'every_positive_weakly_negative_dominated'),r[2])
        cell('S2',count(impossible,'undominated_positive_exists'),r[3])
        cell('S2',f"{st['count']}/{st['denominator']}",r[4]); cell('S2',interval(st),r[5])
    for r in si['S4']:
        data=select('SALAD',r[0],'C20'); cell('S4',len(data),r[1])
        for key,expected in zip(('equal_correct','continuous_oracle','linear_impossible','no_positive','max_correct'),r[2:]): cell('S4',count(data,key),expected)
    for r in si['S5']:
        data=select(r[0],r[1]); counts=Counter(x['interval_shape'] for x in data)
        for key,expected in zip(('empty','interval','disconnected','singleton'),r[2:6]): cell('S5',counts[key],expected)
        widths=[float(x['interval_width']) for x in data if int(x['continuous_oracle'])]
        cell('S5',float(np.median(widths)),r[6])
    for r in si['S6']:
        a=select('SALAD',r[0]); b={x['qid']:x for x in select('Clique',r[0])}
        c=Counter((int(x[r[1]]),int(b[x['qid']][r[1]])) for x in a)
        for key,expected in zip(((1,1),(1,0),(0,1),(0,0)),r[2:]): cell('S6',c[key],expected)
    for r in si['S7']:
        data=select(r[0],r[1],radius=int(r[2])); cell('S7',len(data),r[3])
        for key,expected in zip(('positive_available','equal_correct','continuous_oracle','linear_impossible'),r[4:8]): cell('S7',count(data,key),expected)
        cell('S7',f"{count(data,'eligible_linear_impossible')}/{count(data,'eligible_C_error_C20')}",r[8])
    fixed=rows(out/'robustness/analysis/fixed303_transitions.csv')
    for r in si['S8']:
        st=next(x for x in fixed if (x['representation'],x['policy'],x['radius_m'])==tuple(r[:3]))
        for key,expected in zip(('fixed_10m_cohort','positive_available','equal_correct','continuous_oracle','linear_impossible'),r[3:]): cell('S8',st[key],expected)
    bounded={r['metric']:r for r in rows(out/'robustness/analysis/bounded_statistics.csv')}
    for r in si['S9']:
        st=bounded[r[0]]; cell('S9',f"{st['count']}/303",r[1]); cell('S9',interval(st),r[2])
    contrast=rows(out/'reconstruction/analysis/contrasts.csv')
    direct=rows(out/'reconstruction/analysis/direct_change.csv')
    deletion=rows(out/'reconstruction/analysis/direct_block_deletions.csv')
    for r in si['S10']:
        for quantity,expected in zip(('I4','IDN','J'),r[2:]):
            st=next(x for x in contrast if (x['method'],x['radius_m'],x['quantity'])==(r[0],r[1],quantity))
            cell('S10',f"{float(st['point_pp']):.2f} [{float(st['ci95_low_pp']):.2f}, {float(st['ci95_high_pp']):.2f}]",expected)
    for r in si['S11']:
        st=next(x for x in direct if (x['method'],x['radius_m'])==tuple(r[:2]))
        cell('S11',float(st['point_pp']),r[2]); cell('S11',f"[{float(st['ci95_low_pp']):.2f}, {float(st['ci95_high_pp']):.2f}]",r[3])
    for r in si['S12']:
        data=[float(x['point_pp']) for x in deletion if x['method']==r[0]]
        cell('S12',len(data),r[1]);cell('S12',min(data),r[2]);cell('S12',max(data),r[3])
    for r in si['S13']:
        st=stat(stats,r[0],r[1],'linear_impossible_fraction_of_C_error_C20')
        for key,expected in zip(('denominator','valid_draws','undefined_draws'),r[2:]):cell('S13',st[key],expected)
    for r in si['S14']:
        data=select(r[0],r[1]); route=r[1].split()[0]
        for metric,expected in zip(('linear_headroom','linear_impossible','max_minus_equal'),r[2:]):
            v=[100*np.mean([int(x[metric]) for x in data if x['anchor'] not in block]) for block in draws[route]['blocks']]
            cell('S14',f'{min(v):.2f} to {max(v):.2f}',expected)
    for r in si['S15']:
        data=select(r[0],r[1])
        for a,b,expected in [('equal_r5','max_r5',r[2]),('equal_r20','max_r20',r[3]),('rescue','harm',r[4])]:
            cell('S15',f'{count(data,a)}/{count(data,b)}',expected)
        cell('S15',f"{sum(float(x['equal_distance_m'])>25 for x in data)}/{sum(float(x['max_distance_m'])>25 for x in data)}",r[5])
    for r in si['S16']:
        data=select(r[0],r[1]); st=stat(stats,r[0],r[1],'max_minus_equal')
        near=lambda k:sum(10<float(x[k])<=25 for x in data)
        far=lambda k:sum(float(x[k])>25 for x in data)
        cell('S16',f"{near('equal_distance_m')}/{near('max_distance_m')}",r[2])
        cell('S16',f"{far('equal_distance_m')}/{far('max_distance_m')}",r[3])
        cell('S16',sum(int(x['equal_correct']) and float(x['max_distance_m'])>25 for x in data),r[4])
        cell('S16',f"{count(data,'max_minus_equal')}; {interval(st)}",r[5])
    for r in si['S17']:
        st=next(x for x in rr if x['representation']==r[0] and x['qid']==r[1] and x['policy']=='full' and x['radius_m']=='10')
        for key,expected in zip(('block','positive_count','equal_distance_m','max_distance_m','equal_top1','max_top1'),r[2:]):
            cell('S17',float(st[key]) if key.endswith('_m') else st[key],expected)
    for item, data, fields, selectors in [
        ('S26',cc,('positive_available','equal_correct','scalar_stable','M_stable','D_stable'),lambda r:dict(rep=r[0],radius=r[1],policy='full',offset=2)),
        ('S27',cc,('positive_available','equal_correct','scalar_optimistic','M_optimistic','D_optimistic'),lambda r:dict(rep='SALAD',radius=r[0],policy=r[1],offset=2)),
        ('S29',summary,('positive_available','equal_correct','scalar_optimistic','D_optimistic'),lambda r:dict(rep=r[1],radius=r[2],setting=r[0],offset=4))]:
        for r in si[item]:
            sel=selectors(r); offset=sel.pop('offset'); rep=sel.pop('rep')
            for metric,expected in zip(fields,r[offset:]):
                st=stat(data,rep,'SNU morning_four',metric,**sel);cell(item,st['count'],expected)
                if item=='S29':cell(item,st['queries'],r[3])
    for r in si['S28']:
        data={x['metric']:x['count'] for x in overlaps if x['pair'].replace('_vs_','/')==r[0] and x['radius_m']==r[1]
              and x['route']=='SNU' and x['group']=='morning_four'}
        for key,expected in zip(('both_correct','a_only','b_only','both_wrong','same_wrong_candidate'),r[2:]):cell('S28',data[key],expected)
    block=rows(out/'capability/analysis/block_sensitivity.csv')
    for r in si['S30']:
        st=next(x for x in block if x['route']=='SNU' and x['representation']=='SALAD' and x['group']=='morning_four'
                and x['metric']=='D_stable' and (x['radius_m'],x['raw_block_length'])==tuple(r[:2]))
        cell('S30',st['blocks'],r[2]);cell('S30',st['count'],r[3]);cell('S30',f"{float(st['ci95_low']):.2f} to {float(st['ci95_high']):.2f}",r[4])
    pops=[('SNU','morning_four'),('SNU','day_night'),('Valley','all_cross'),('NSAVP','all_cross')]
    for r in si['S31']:
        arm={'Strictly monotone (M)':'monotone'}.get(r[0],r[0])
        for (route,name),expected in zip(pops,r[2:]):
            st=next(x for x in seeds if (x['arm'],x['seed'],x['route'],x['group'],x['radius_m'])==(arm,r[1],route,name,'10'))
            cell('S31',float(st['R1_percent']),expected)
    decision=read(out/'capability/training/DECISION.json')
    for r in si['S32']:
        arm={'Strictly monotone (M)':'monotone'}.get(r[0],r[0])
        st=next(x for x in decision['arms'] if x['arm']==arm)
        computed=[f"{st['SNU_morning_gain_pp']:.2f}",str(sum(v>0 for v in st['directional_gain_pp'])),
                  f"{st['DayNight_gain_pp']:.2f}",f"{st['Valley_cross_gain_pp']:.2f}",
                  f"{st['NSAVP_net_gain_queries']:.2f}",'Pass' if st['passed'] else 'Fail']
        for a,b in zip(computed,r[1:]):cell('S32',a,b)
    for r,arm in zip(main['2'],('scalar','monotone','unconstrained')):
        for (route,name),expected in zip(pops,r[2:]):
            values=[float(x['R1_percent']) for x in seeds if (x['arm'],x['route'],x['group'],x['radius_m'])==(arm,route,name,'10')]
            require(len(values)==3,'Three seeds required')
            cell('Main Table 2',r'\('+f'{np.mean(values):.2f}'+r'\pm'+f'{np.std(values,ddof=1):.2f}'+r'\)',expected)
    # Official Top-20 correctness is recalculated without invoking the native model.
    official=rows(out/'capability/inputs/official_top20.csv')
    official_scores=dict(np.load(out/'capability/inputs/scores.npz',allow_pickle=False))
    official_geo=np.load(out/'capability/inputs/geography.npz',allow_pickle=False)['geo']
    official_frames=read(out/'capability/inputs/frame_metadata.json')['frames']
    query_indices={str(q):i for i,q in enumerate(official_scores['qids'])};joins={}
    for record in official:
        qid=f"{record['location']}:{record['condition']}:{record['anchor_id']}"
        if qid not in joins:
            qi=query_indices[qid];n=int(official_scores['lengths'][qi])
            joins[qid]={str(official_frames[int(bank)]['original_id']):float(official_geo[qi,j])
                        for j,bank in enumerate(official_scores['ids'][qi,:n])}
            require(len(joins[qid])==n,'Official original-frame identity collision')
        distance=joins[qid][record['database_id']]
        require(abs(distance-float(record['geographic_distance_m']))<1e-7,'Official geographic identity mismatch')
        record['geographic_distance_m']=str(distance)
    for r in si['S20']:
        data=[x for x in official if (x['location'],x['condition'])==tuple(r[:2])]
        byid=defaultdict(list)
        for x in data:byid[x['anchor_id']].append(x)
        cell('S20',len(byid),r[2])
        for k,expected in zip((1,5,20),r[3:]):
            n=sum(any(int(x['rank'])<=k and float(x['geographic_distance_m'])<=10 for x in values) for values in byid.values())
            cell('S20',100*n/len(byid),expected)
    for r,(rep,metric) in zip(main['1'], [('SALAD','equal_correct'),('Official','official'),('SALAD','max_correct'),('Clique','equal_correct'),('Clique','max_correct')]):
        for (route,name),expected in zip(pops,r[2:]):
            if rep=='Official':
                if route=='NSAVP':cell('Main Table 1','NA',expected);continue
                data=[x for x in official if x['location']==route and int(x['rank'])==1
                      and (group(dict(route=route,condition=x['condition']))==name or name=='all_cross' and group(dict(route=route,condition=x['condition']))!='same')]
                value=100*sum(float(x['geographic_distance_m'])<=10 for x in data)/len(data)
            else:
                data=select(rep,route+' '+name);value=100*count(data,metric)/len(data)
            cell('Main Table 1',value,expected)
    figure_checks=verify_figure_data(root,out,rr,stats,contrast,direct)
    inventory=[]
    historical={'S18','S19','S21','S22','S23','S24','S25','S33'}
    bindings=read(root/'SOURCE_BINDINGS.json')
    for label in ['Main Table 1','Main Table 2',*[f'S{i}' for i in range(1,34)]]:
        documentary=label in historical
        require(documentary or checks[label]>0, 'Unverified displayed table: '+label)
        inventory.append(dict(item=label,mode='SOURCE_VERIFICATION_ONLY' if documentary else 'RECALCULATED_DISPLAY_CELLS',
                              checked_cells=checks[label],status='SOURCE_BOUND' if documentary else 'PASSED',
                              source_files=bindings.get(label,{}),
                              limitation='Historical control, execution, or qualification record; no matching/training/inference rerun' if documentary else ''))
    for label in [*[f'Figure{i}' for i in range(1,6)],*[f'FigureS{i}' for i in range(1,7)]]:
        documentary=label in ('Figure5','FigureS4','FigureS5')
        inventory.append(dict(item=label,mode='PIXEL_SOURCE_VERIFICATION_ONLY' if documentary else 'RECALCULATED_PLOT_DATA',
                              checked_cells=figure_checks.get(label,0),status='SOURCE_BOUND' if documentary else 'PASSED',
                              source_files=bindings.get(label,{}),
                              limitation='Fixed source-image identity; redistribution authorization pending' if documentary else ''))
    require(len(inventory)==46,'46 display items required')
    write(out/'DISPLAY_COVERAGE.json',inventory)
    write(out/'DISPLAY_CHECKS.json',dict(table_cells=sum(checks.values()),figure_checks=figure_checks,
          historical_items=sorted(historical),source_only_items_are_not_full_reruns=True))
    return len(inventory)


def verify_figure_data(root,out,rr,stats,contrast,direct):
    from reproduce import compare_value, require, read, write
    def cmp(label,a,b):compare_value(a,b,label)
    fig1=read(root/'reference/figures/Figure1_plotted.json')
    bands=rows(out/'reconstruction/analysis/bands.csv')
    selected=[next(r for r in bands if (r['route'],r['group'],r['method'])==('SNU','morning_four',m)) for m in fig1['methods']]
    cmp('Figure1 bands',[[int(r[k]) for k in ('le10','10to25','gt25')] for r in selected],fig1['distance_count_matrix'])
    cmp('Figure1 recall',[[int(r['le10'])/532*100,float(r['R_at20_10m_percent'])] for r in selected],fig1['recall_R1_R20'])
    partitions=rows(out/'capability/analysis/partitions.csv')
    fig2=read(root/'reference/figures/Figure2_current.json')
    classes=('EQUAL_CORRECT','SCALAR_OPPORTUNITY','NONLINEAR_D_OPPORTUNITY','STRICT_BOTH_OBSTRUCTION')
    computed=[[int(next(r['count'] for r in partitions if r['representation']==rep and r['radius_m']=='10' and r['classification']==cls)) for cls in classes] for rep in ('SALAD','Clique')]
    cmp('Figure2 partition',computed,fig2['counts'])
    capacities=rows(out/'capability/analysis/statistics.csv')
    def capacity(radius,policy,metric):
        found=[r for r in capacities if (r['representation'],r['route'],r['group'],r['radius_m'],r['policy'],r['metric'])
               ==('SALAD','SNU','morning_four',str(radius),policy,metric)]
        require(len(found)==1,'Current figure selector')
        return found[0]
    cmp('Figure2 radius curves',[[float(capacity(radius,'full',metric)['count']) for metric in fig2['metrics']]
                                 for radius in fig2['radius_m']],fig2['radius_counts'])
    fig3=read(root/'reference/figures/Figure3_plotted.json')
    computed=[[[float(next(r[k] for r in contrast if (r['method'],int(r['radius_m']),r['quantity'])==(m,radius,'J'))) for k in ('point_pp','ci95_low_pp','ci95_high_pp')] for radius in (5,10,15,20,25)] for m in ('C','Clique_C')]
    cmp('Figure3 contrasts',computed,fig3['estimates_point_low_high'])
    cmp('Figure3 paired changes',[[float(next(r[k] for r in direct if r['method']==m and r['comparison']=='J25minus10')) for k in ('point_pp','ci95_low_pp','ci95_high_pp')] for m in ('C','Clique_C')],fig3['direct_point_low_high'])
    fig4=read(root/'reference/figures/Figure4_plotted.json')
    curves=[]
    for rep in ('SALAD','Clique'):
        curves.append([[sum(int(r[k]) for r in rr if r['representation']==rep and r['policy']=='full' and r['route']=='SNU' and group(r)=='morning_four' and int(r['radius_m'])==radius)/532*100 for k in ('equal_correct','continuous_oracle')] for radius in (5,10,15,20,25)])
    cmp('Figure4 curves',curves,fig4['curve_percent'])
    fixed=rows(out/'robustness/analysis/fixed303_transitions.csv')
    f=[next(r for r in fixed if r['representation']=='SALAD' and r['policy']=='full' and int(r['radius_m'])==radius) for radius in (5,10,15,20,25)]
    cmp('Figure4 fixed303',[[int(r['equal_correct']),int(r['continuous_oracle'])-int(r['equal_correct']),int(r['linear_impossible']),303-int(r['positive_available'])] for r in f],fig4['fixed303_counts'])
    figS1=read(root/'reference/figures/FigureS1_plotted.json')
    for record in figS1['rows']:
        data=[r for r in rr if r['representation']==record['representation'] and r['route']==record['route'] and r['policy']=='full' and r['radius_m']=='10'
              and (group(r)==record['group'] or record['group']=='all_cross' and group(r)!='same')]
        st=next(r for r in stats if (r['representation'],r['route'],r['group'],r['policy'],r['radius_m'],r['metric'])==(record['representation'],record['route'],record['group'],'full','10','max_minus_equal'))
        for key,a in [('delta_R1_pp',float(st['value_percent'])),('paired_spatial_ci95_low',float(st['ci95_low'])),('paired_spatial_ci95_high',float(st['ci95_high'])),
                      ('rescues',sum(int(x['rescue']) for x in data)),('harms',sum(int(x['harm']) for x in data)),('queries',len(data))]:cmp('FigureS1/'+key,a,record[key])
    scores=dict(np.load(root/'payload/capability/inputs/scores.npz',allow_pickle=False))
    geo=np.load(root/'payload/capability/inputs/geography.npz',allow_pickle=False)['geo']
    qindex={str(q):i for i,q in enumerate(scores['qids'])}
    case_rows=rows(root/'reference/figures/FigureS2_plotted.csv')
    for record in case_rows:
        qi=qindex[record['qid']]; n=int(scores['lengths'][qi]); ix=np.flatnonzero(scores['ids'][qi,:n]==int(record['candidate_id']))
        require(len(ix)==1,'FigureS2 candidate identity');j=int(ix[0])
        for key,a in [('rgb_squared_distance',float(scores['dr_s'][qi,j])),('thermal_squared_distance',float(scores['dt_s'][qi,j])),('published_distance_m',float(geo[qi,j])),('positive_10m',int(geo[qi,j]<=10))]:cmp('FigureS2/'+key,a,float(record[key]))
    figS3=read(root/'reference/figures/FigureS3_current.json')
    for record in figS3:
        cmp('FigureS3 candidate capacities',float(capacity(record['radius_m'],record['policy'],record['metric'])['value_percent']),record['value_percent'])
    figS6=read(root/'reference/figures/FigureS6_current.json')
    seeds=rows(out/'capability/training/seed_summary.csv')
    for record in figS6:
        found=[r for r in seeds if all(r[k]==record[k] for k in ('arm','seed','route','group','radius_m'))]
        require(len(found)==1,'FigureS6 seed identity')
        cmp('FigureS6 seed gain',float(found[0]['gain_pp']),float(record['gain_pp']))
    write(out/'PLOT_DATA.json',dict(Figure1=fig1,Figure2=fig2,Figure3=fig3,Figure4=fig4,FigureS3=figS3,FigureS6=figS6,
          note='Values independently reconstructed and checked before these display specifications were exported'))
    return dict(Figure1=20,Figure2=23,Figure3=36,Figure4=40,FigureS1=48,FigureS2=len(case_rows)*4,
                FigureS3=len(figS3),FigureS6=len(figS6))
