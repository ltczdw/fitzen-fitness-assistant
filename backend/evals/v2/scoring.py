"""Correct denominators and held-out threshold calibration, independent of app."""
from collections import Counter, defaultdict


def distribution(values):
    """MEWHELP-style score distribution; no model-written summary."""
    values=sorted(values)
    if not values:return {'samples':0,'min':None,'p25':None,'p50':None,'p75':None,'max':None}
    def p(q):return values[min(len(values)-1,int(q*len(values)))]
    return {'samples':len(values),'min':values[0],'p25':p(.25),'p50':p(.5),'p75':p(.75),'max':values[-1]}


def rag_metrics(rows):
    rows=[r for r in rows if r['category']=='rag' and 'gold_answerable' in r.get('actual',{})]
    positive=[r['actual'] for r in rows if r['actual']['gold_answerable']]
    negative=[r['actual'] for r in rows if not r['actual']['gold_answerable']]
    def mean(values):return sum(values)/len(values) if values else None
    return {'samples':len(rows),'answerable_samples':len(positive),'refusal_samples':len(negative),
            'answerable_acceptance':mean([bool(a['answerable']) for a in positive]),
            'unanswerable_refusal':mean([not a['answerable'] for a in negative]),
            'evidence_recall_at_4':mean([a['evidence_recall'] for a in positive]),
            # Every answerable query stays in the denominator, including zero-hit queries.
            'mrr':mean([a['mrr'] for a in positive]),
            'citation_precision':mean([a['citation_precision'] or 0.0 for a in positive]),
            'generation_evaluated':False,'answer_faithfulness':None}


def calibrate(rows):
    """Suggest a threshold on calibration rows only, separately for each mode.

    Default gate still decides acceptance. Never apply a fitted threshold to
    test rows while fitting, never write it into app Settings.
    """
    groups=defaultdict(list)
    for row in rows:
        actual=row.get('actual',{})
        if row['category']=='rag' and row['split']=='calibration' and 'confidence' in actual:
            groups[actual['mode']].append(actual)
    output={}
    for mode,items in groups.items():
        if mode=='none':
            output[mode]={'status':'not_calibratable_no_evidence','samples':len(items)}
            continue
        if not any(i['gold_answerable'] for i in items) or all(i['gold_answerable'] for i in items):
            output[mode]={'status':'insufficient_class_coverage','samples':len(items)}
            continue
        stats={'answerable':distribution([i['confidence'] for i in items if i['gold_answerable']]),
               'unanswerable':distribution([i['confidence'] for i in items if not i['gold_answerable']])}
        scan=[];best=None
        for step in range(1,100):
            threshold=step/100
            positive=[i for i in items if i['gold_answerable']]
            negative=[i for i in items if not i['gold_answerable']]
            # Retain per-task stricter offset, rather than flattening safety_triage.
            accepts=lambda i:i['confidence']>=min(.95,threshold+max(0,i['threshold']-i.get('base_threshold',.48)))
            tpr=sum(accepts(i) for i in positive)/len(positive)
            tnr=sum(not accepts(i) for i in negative)/len(negative)
            score=(tpr+tnr)/2
            candidate=(score,tnr,threshold)
            scan.append({'threshold':threshold,'answerable_acceptance':tpr,'unanswerable_refusal':tnr,'youden_j':tpr+tnr-1})
            if best is None or candidate>best[0]:best=(candidate,{'threshold':threshold,'balanced_accuracy':score,'answerable_acceptance':tpr,'unanswerable_refusal':tnr})
        output[mode]={'status':'suggestion_only','samples':len(items),**best[1],
                      'score_distribution':stats,'scan':scan,
                      'base_thresholds_in_use':sorted({i.get('base_threshold',.48) for i in items}),
                      'test_used_for_calibration':False}
    return output


def summarize(rows):
    bycategory=defaultdict(list)
    for r in rows:bycategory[r['category']].append(r)
    critical=[r for r in rows if r['severity']=='critical']
    counts=Counter(r['status'] for r in rows)
    summary={'counts':dict(counts),'scenarios':len(rows),'families':len({r['family'] for r in rows}),
             'critical_failures':sum(r['status']!='passed' for r in critical),
             'categories':{k:{'passed':sum(r['status']=='passed' for r in v),'total':len(v),
                              'pass_rate':sum(r['status']=='passed' for r in v)/len(v)} for k,v in bycategory.items()},
             'rag_default_by_split':{s:rag_metrics([r for r in rows if r['split']==s]) for s in ('development','calibration','test')},
             'calibration':calibrate(rows), 'default_acceptance':{}}
    test=[r for r in rows if r['split']=='test']
    normal=[r for r in test if r['severity']!='critical' and r['category']!='rag']
    rate=sum(r['status']=='passed' for r in normal)/len(normal) if normal else None
    ragtest=summary['rag_default_by_split']['test']
    summary['default_acceptance']={
        'critical_zero_failures':summary['critical_failures']==0,
        'normal_test_success_at_least_0.90':None if rate is None else rate>=.90,
        'rag_answerable_acceptance_at_least_0.85':None if ragtest['answerable_acceptance'] is None else ragtest['answerable_acceptance']>=.85,
        'rag_unanswerable_refusal_at_least_0.95':None if ragtest['unanswerable_refusal'] is None else ragtest['unanswerable_refusal']>=.95,
    }
    # An execution exception is never treated as a legitimate refusal or missing sample.
    gates=list(summary['default_acceptance'].values())
    summary['release_gate_passed']=all(g is True for g in gates) and not counts.get('error',0)
    summary['complete_acceptance_coverage']=all(g is not None for g in gates)
    return summary
