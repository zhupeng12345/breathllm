"""对三折三种子 T03 归档执行只读 A/B/Q 事件归因。"""

import argparse
import importlib
import io
import json
import signal
from pathlib import Path
from time import monotonic

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

r0 = importlib.import_module('BreathLLM-R0')
import 发布复现 as repro

audit, b07, track = r0.audit, r0.b07, r0.track
CLASSES = tuple(b07.CLASS_NAMES)
assert CLASSES == ('Normal','OSA','CSA','MSA','Hypopnea')
QUANTILES = (.1,.25,.5,.75,.9)
BOOT, SEED, TOLERANCE = 10000, 74121, 1e-6
OUT = repro.OUTPUT/'BreathLLM-R2结果.json'
FAILURE = repro.OUTPUT/'BreathLLM-R2未完成.json'
ERRORS = ('unmatched','q_wrong','b_blocks_correct_q','hyp_loses_to_pause','correct')


def max_abs(x):
    return float(np.abs(x).max(initial=0))


def decompose(p):
    p = p.astype(np.float64)
    a,b,q = r0.branches(p)
    pause = p[:,1:4].sum(1)
    vb = np.isfinite(b)
    vq = np.isfinite(q).all(1)
    ab = np.full_like(a,np.nan)
    h = np.full_like(a,np.nan)
    q_mass = np.full_like(q,np.nan)
    ab[vb] = a[vb]*b[vb]
    h[vb] = a[vb]*(1-b[vb])
    usable = vb & vq
    q_mass[usable] = ab[usable,None]*q[usable]
    # A zero pause mass fixes all pause class masses even though q is undefined.
    q_mass[vb & ~vq & (pause==0)] = 0.
    errors = dict(normal=max_abs((1-a)-p[:,0]),pause=max_abs(ab[vb]-pause[vb]),
                  hyp=max_abs(h[vb]-p[vb,4]),q_mass=max_abs(q_mass[usable]-p[usable,1:4]))
    assert max(errors.values()) < TOLERANCE
    invalid = dict(b=int((~vb).sum()),q=int((~vq).sum()),small_b_range_excursions=int((vb & ((b<0)|(b>1))).sum()))
    return a,b,q,h,q_mass,errors,invalid


def interval_sums(values, alarms, positive):
    csum = np.vstack((np.zeros(values.shape[1]),np.cumsum(values*positive[:,None],axis=0)))
    return csum[alarms[:,1]]-csum[alarms[:,0]]


def reproduce(y,p,clock,archived,receipt_counts,a,h,qa):
    official,positive,matches = b07.event_score(y,p)
    native, native_matches = track.totals(y,p,.5,**b07.OPTIONS)
    np.testing.assert_array_equal(matches,native_matches)
    np.testing.assert_array_equal(positive,(y>=0)&(p[:,1:].sum(1)>=.5))
    audit.check_matches(y,positive,matches)
    assert official==archived['counts']
    assert all(official[k]==v for k,v in native.items())
    assert all(official[k]==receipt_counts[k] for k in official)
    alarms = b07.runs(positive.astype(int),gap=b07.OPTIONS['gap'],min_positive=b07.OPTIONS['min_run'])
    matched = np.isfinite(matches[:,3])
    ids = np.searchsorted(alarms[:,0],(matches[matched,0]+matches[matched,4]).astype(int))
    original = interval_sums(p[:,1:],alarms,positive)
    native_pred = original.argmax(1)+1
    np.testing.assert_array_equal(native_pred[ids],matches[matched,3])
    np.testing.assert_array_equal(np.bincount(native_pred,minlength=5)[1:],official['alarms_by_cls'])
    saved_truth = np.asarray([r['reference']+[r['label']] for r in archived['events']],dtype=int).reshape(-1,3)
    np.testing.assert_array_equal(matches[:,:3].astype(int),saved_truth-[clock[0],clock[0],0])
    saved = [r for r in archived['events'] if r['matched']]
    np.testing.assert_array_equal(matched,[r['matched'] for r in archived['events']])
    np.testing.assert_array_equal(alarms[ids,:2]+clock[0],np.asarray([r['alarm'] for r in saved],dtype=int).reshape(-1,2))
    np.testing.assert_array_equal(native_pred[ids],[r['prediction'] for r in saved])
    np.testing.assert_array_equal(original[ids],np.asarray([r['scores'][1:] for r in saved]).reshape(-1,4))
    assert np.isfinite(np.column_stack((a,qa,h))[positive]).all(), '正式阳性位置分母无效'
    # Production uses the positive-second indicator as the only event weight.
    evidence = interval_sums(np.where(positive[:,None],np.column_stack((a,qa,h)),0.),alarms,positive)
    mass_error = max_abs(evidence[:,0]-evidence[:,1:].sum(1))
    assert mass_error<TOLERANCE
    np.testing.assert_array_equal(evidence[:,1:].argmax(1)+1,native_pred,
                                  err_msg='FP64 分解未能逐报警复现原始分型')
    return official,positive,matches,alarms,dict(zip(np.flatnonzero(matched),ids)),evidence,original,native_pred,mass_error


def binary_scores(label,scores):
    valid = np.isfinite(scores)
    label,scores = np.asarray(label)[valid],np.asarray(scores)[valid]
    n_pos = int(label.sum())
    n_neg = len(label)-n_pos
    both = n_pos>0 and n_neg>0
    return dict(auroc=float(roc_auc_score(label,scores)) if both else None,
                auprc=float(average_precision_score(label,scores)) if both else None,
                positive=n_pos,negative=n_neg,invalid=int((~valid).sum()))


def quantile(x,level=QUANTILES):
    return np.quantile(x,level).tolist() if len(x) else None


def longest_true(x):
    idx = np.flatnonzero(np.diff(np.r_[False,x,False]))
    return int(np.diff(idx.reshape(-1,2),axis=1).max(initial=0))


def detection_features(left,right,a,positive,alarms,alarm_id,origin):
    a_ref = a[left:right]
    actual = positive[left:right]
    overlap = np.maximum(0,np.minimum(right,alarms[:,1])-np.maximum(left,alarms[:,0]))
    distance = np.maximum(0,np.maximum(left-alarms[:,1],alarms[:,0]-right))
    nearest = int(np.lexsort((alarms[:,0],-overlap,distance))[0]) if len(alarms) else None
    p10,median,p90 = quantile(a_ref,(.1,.5,.9))
    return dict(a_mean=float(a_ref.mean()),a_median=median,a_p10=p10,a_p90=p90,
                a_ge_half_seconds=int((a_ref>=.5).sum()),production_positive_seconds=int(actual.sum()),
                longest_a_ge_half=longest_true(a_ref>=.5),longest_production_positive=longest_true(actual),
                matched_overlap_seconds=int(overlap[alarm_id]) if alarm_id is not None else 0,
                best_alarm_overlap_seconds=int(overlap.max(initial=0)),
                nearest_alarm_distance_seconds=int(distance[nearest]) if nearest is not None else None,
                nearest_alarm=(alarms[nearest,:2]+origin).tolist() if nearest is not None else None)


def classify_event(index,reference,a,positive,alarms,match_ids,ev,raw,official,origin):
    left,right,label = map(int,reference[:3])
    alarm_id = match_ids.get(index)
    row = dict(start=left+origin,stop=right+origin,duration=right-left,label=label,
               **detection_features(left,right,a,positive,alarms,alarm_id,origin))
    if alarm_id is None:
        return dict(row,error='unmatched',prediction=0)
    A,*scores = ev[alarm_id]
    Q,H = np.asarray(scores[:3]), scores[3]
    q_choice = int(Q.argmax()+1)
    prediction = int(official[alarm_id])
    margin = H-Q.max() if label==4 else Q[label-1]-max(H,np.delete(Q,label-1).max())
    q_error = label<4 and q_choice!=label
    b_blocks = label<4 and q_choice==label and H>Q[label-1]
    if prediction==label:
        error = 'correct'
    elif label==4:
        error = 'hyp_loses_to_pause'
    else:
        error = 'q_wrong' if q_error else 'b_blocks_correct_q'
        assert q_error or b_blocks
    start,stop = alarms[alarm_id,:2]
    return dict(row,alarm=(alarms[alarm_id,:2]+origin).tolist(),prediction=prediction,error=error,
                weighted_positive_seconds=int(positive[start:stop].sum()),A=float(A),H=float(H),Q=Q.tolist(),
                production_scores=raw[alarm_id].tolist(),q_argmax=q_choice if Q.sum()>0 else None,
                b_event=float(Q.sum()/A) if A>0 else None,margin=float(margin),
                b_competition=float(H-Q.max()),q_wrong=bool(q_error),b_blocks_correct_q=bool(b_blocks),
                all_pause_lose_to_hyp=bool(H>Q.max()))


def matrix_metrics(cm):
    c = np.asarray(cm)
    support = c.sum(1)
    pred = c.sum(0)
    den = support+pred
    f1 = np.divide(2*c.diagonal(),den,out=np.zeros(len(c)),where=den>0)
    return dict(cm=c.tolist(),macro_f1=float(f1.mean()) if c.sum() else None,
                supported_macro_f1=float(f1[support>0].mean()) if c.sum() else None,
                recall=[float(n/d) if d>0 else None for n,d in zip(c.diagonal(),support,strict=True)],
                support=support.tolist())


def q_metrics(y,q,argmax=False):
    if argmax:
        m = (y>0)&(y<4)&(q>0)
        guess = q[m]-1
    else:
        m = (y>0)&(y<4)&np.isfinite(q).all(1)
        guess = q[m].argmax(1)
    truth = y[m]-1
    c = np.bincount(3*truth+guess,minlength=9).reshape(3,3)
    return dict(**matrix_metrics(c),invalid_true_pause=int(((y>0)&(y<4)&~m).sum()))


def branch_metrics(y,a,b,q,events):
    normal = y==0
    abnormal = y>0
    hyp = y==4
    apnea = abnormal & ~hyp
    seconds = dict(a_abnormal_vs_normal=binary_scores(abnormal[abnormal|normal],a[abnormal|normal]),
                   a_hyp_vs_normal=binary_scores(hyp[hyp|normal],a[hyp|normal]),
                   b_apnea_vs_hyp=binary_scores(apnea[abnormal],b[abnormal]),
                   q=q_metrics(y,q))
    # Each reference event gets one mean a. Natural uninterrupted valid Normal
    # runs provide the event-level negative controls without changing the scorer.
    normal_spans = b07.runs(normal.astype(int))
    norm_values = [float(a[left:right].mean()) for left,right,_ in normal_spans]
    is_hyp = [row['a_mean'] for row in events if row['label']==4]
    def event_a(positive):
        return binary_scores(np.arange(len(positive)+len(norm_values))<len(positive),np.r_[positive,norm_values])
    matched = [row for row in events if row['prediction']>0]
    true = np.array([row['label'] for row in matched],dtype=int)
    b_event = np.array([row['b_event'] for row in matched],dtype=float)
    q_class = np.array([row['q_argmax'] or 0 for row in matched],dtype=int)
    event = dict(a_abnormal_vs_normal=event_a([row['a_mean'] for row in events]),
                 a_hyp_vs_normal=event_a(is_hyp),
                 b_apnea_vs_hyp=binary_scores(true<4,b_event),q=q_metrics(true,q_class,argmax=True))
    return dict(seconds=seconds,events=event)


def one_night(y,p,clock,prior,receipt):
    a,b,q,h,qa,identities,invalid = decompose(p)
    result = reproduce(y,p,clock,prior,receipt,a,h,qa)
    counts,pos,matches,alarms,match_ids,ev,raw,pred,a_error = result
    # All equivalence checks above precede any branch attribution.
    events = [classify_event(i,row,a,pos,alarms,match_ids,ev,raw,pred,int(clock[0]))
              for i,row in enumerate(matches)]
    disjoint = np.zeros((4,5),dtype=int)
    np.add.at(disjoint,([r['label']-1 for r in events],[r['prediction'] for r in events]),1)
    np.testing.assert_array_equal(disjoint,counts['event_cm'])
    false_alarm_ids = sorted(set(range(len(alarms)))-set(match_ids.values()))
    all_alarms = np.column_stack((alarms[:,:2]+clock[0],pred))
    return dict(counts=counts,events=events,alarms=all_alarms.tolist(),unmatched_alarm_indices=false_alarm_ids,
                metrics=branch_metrics(y,a,b,q,events),identities=dict(**identities,event_mass=a_error),
                invalid=invalid,positive_sha256=audit.digest(pos.tobytes()),
                fp64_a_threshold_disagreements=int(((y>=0)&(a>=.5)!=pos).sum()),
                max_alarm_score_difference=max_abs(ev[:,1:]-raw))


DETECTION_FIELDS = ('a_mean','a_median','a_p10','a_p90','a_ge_half_seconds',
                    'production_positive_seconds','longest_a_ge_half','longest_production_positive',
                    'matched_overlap_seconds','best_alarm_overlap_seconds','nearest_alarm_distance_seconds')


def event_stats(events):
    output = {}
    for label,name in enumerate(CLASSES[1:],1):
        subset = [row for row in events if row['label']==label]
        n = len(subset)
        errors = {kind:sum(row['error']==kind for row in subset) for kind in ERRORS}
        groups = {'all':subset,'matched':[r for r in subset if r['prediction']>0],
                  'unmatched':[r for r in subset if r['prediction']==0]}
        detection = {}
        for status,rows in groups.items():
            detection[status] = {}
            for field in DETECTION_FIELDS:
                vector = [row[field] for row in rows if row[field] is not None]
                detection[status][field] = dict(n=len(vector),mean=float(np.mean(vector)) if vector else None,
                                                quantiles=quantile(vector))
        margin = np.array([row['margin'] for row in groups['matched']])
        misses = groups['unmatched']
        flags = dict(zero_positive=sum(r['production_positive_seconds']==0 for r in misses),
                     fewer_than_five_positive=sum(r['production_positive_seconds']<5 for r in misses),
                     no_overlap=sum(r['best_alarm_overlap_seconds']==0 for r in misses),
                     all_pause_lose_to_hyp=sum(r['all_pause_lose_to_hyp'] for r in groups['matched']),
                     q_wrong_and_all_pause_lose=sum(r['q_wrong'] and r['all_pause_lose_to_hyp'] for r in groups['matched']))
        output[name] = dict(n=n,matched=len(margin),errors=errors,flags=flags,
                            rates={kind:count/n if n else None for kind,count in errors.items()},
                            transitions=[sum(r['prediction']==i for r in subset) for i in range(5)],
                            margins=dict(negative=int((margin<0).sum()),zero=int((margin==0).sum()),
                                         positive=int((margin>0).sum()),quantiles=quantile(margin),
                                         mean=float(margin.mean()) if len(margin) else None),
                            detection=detection)
    return output


def metric_values(stats, measured):
    values = {}
    for unit, metrics in measured.items():
        for contrast, row in metrics.items():
            for field in ('auroc','auprc','supported_macro_f1'):
                if field in row:
                    values[f'{unit}/{contrast}/{field}'] = row[field]
            if contrast == 'q':
                for name,v in zip(CLASSES[1:4],row['recall'],strict=True):
                    values[f'{unit}/q/recall/{name}'] = v
    for name,g in stats.items():
        for error,rate in g['rates'].items():
            values[f'{name}/rate/{error}'] = rate
            if error!='unmatched':
                values[f'{name}/matched/rate/{error}'] = g['errors'][error]/g['matched'] if g['matched'] else None
        values[f'{name}/matched/margin_mean'] = g['margins']['mean']
        values[f'{name}/unmatched/zero_positive_rate'] = g['flags']['zero_positive']/g['errors']['unmatched'] if g['errors']['unmatched'] else None
        values[f'{name}/unmatched/fewer_than_five_positive_rate'] = g['flags']['fewer_than_five_positive']/g['errors']['unmatched'] if g['errors']['unmatched'] else None
        for status,fields in g['detection'].items():
            for field,data in fields.items():
                values[f'{name}/{status}/{field}'] = data['mean']
    return values


def interval(vector):
    good = vector[np.isfinite(vector)]
    return dict(ci95=np.quantile(good,[.025,.975]).tolist() if len(good) else None,
                valid=int(len(good)),valid_fraction=len(good)/BOOT)


def mean_defined(values):
    good = [x for x in values if x is not None]
    return float(np.mean(good)) if good else None


def top_contribution(vector,names):
    total = float(np.sum(vector))
    ranked = np.argsort(-vector,kind='stable')[:5]
    return dict(events=total,subjects=int((vector>0).sum()),
                top5=[dict(person=names[i],events=float(vector[i])) for i in ranked if vector[i]>0],
                top5_share=float(vector[ranked].sum()/total) if total else None)


def f1_scores(t):
    denom = t['truth']+t['predicted']
    return dict(event_f1=2*t['detected']/denom,typed_event_f1=2*t['typed']/denom,
                fa_h=3600*(t['predicted']-t['detected'])/t['seconds'])


def repair_gains(t,cases):
    denom = t['truth']+t['predicted']
    base = f1_scores(t)['typed_event_f1']
    # Typed correction at fixed alarms vs one extra correctly typed report per miss.
    return {name:(2*(t['typed']+count)/(denom+count)-base if name.endswith('_misses') else
                  (2*t['typed']/(denom-count)-base if name=='false_alarms' else 2*count/denom))
            for name,count in cases.items()}


def error_counts(rows,counts):
    return dict(hyp_misses=rows['Hypopnea']['errors']['unmatched'],
                apnea_misses=sum(rows[c]['errors']['unmatched'] for c in CLASSES[1:4]),
                all_detection_misses=counts['truth']-counts['detected'],
                q_wrong=sum(rows[c]['errors']['q_wrong'] for c in CLASSES[1:4]),
                b_blocks_correct_q=sum(rows[c]['errors']['b_blocks_correct_q'] for c in CLASSES[1:4]),
                hyp_loses_to_pause=rows['Hypopnea']['errors']['hyp_loses_to_pause'],
                false_alarms=counts['predicted']-counts['detected'])


def analysis(people,limit):
    names = list(people)
    draws = np.random.default_rng(SEED).integers(len(names),size=(BOOT,len(names)))
    averaged,person_cases,person_counts,contribution = {},[],[],{}
    q_nights = {unit:[] for unit in ('seconds','events')}
    for person,nights in people.items():
        limit()
        values,count_per_seed,case_per_seed = [],[],[]
        for night in nights.values():
            stats = event_stats(night['events'])
            night['by_class'] = stats
            values.append(metric_values(stats,night['metrics']))
            count_per_seed.append({k:night['counts'][k] for k in ('seconds','truth','predicted','detected','typed')})
            case_per_seed.append(error_counts(stats,night['counts']))
        averaged[person] = {k:mean_defined([v[k] for v in values]) for k in values[0]}
        count_avg = {k:mean_defined([v[k] for v in count_per_seed]) for k in count_per_seed[0]}
        case_avg = {k:mean_defined([v[k] for v in case_per_seed]) for k in case_per_seed[0]}
        person_counts.append(count_avg)
        person_cases.append(case_avg)
        for unit in q_nights:
            q_nights[unit].append(np.mean([r['metrics'][unit]['q']['cm'] for r in nights.values()],axis=0))
        for name in CLASSES[1:]:
            summary = [r['by_class'][name] for r in nights.values()]
            contribution.setdefault(f'{name}/truth',[]).append(np.mean([r['n'] for r in summary]))
            contribution.setdefault(f'{name}/matched',[]).append(np.mean([r['matched'] for r in summary]))
            for error in ERRORS:
                contribution.setdefault(f'{name}/{error}',[]).append(np.mean([r['errors'][error] for r in summary]))
    stats = {}
    for key in next(iter(averaged.values())):
        subjects = [v[key] for v in averaged.values()]
        vector = np.array(subjects,dtype=float)
        sampled = vector[draws]
        valid = np.isfinite(sampled).sum(1)
        means = np.divide(np.nansum(sampled,axis=1),valid,out=np.full(BOOT,np.nan),where=valid>0)
        stats[key] = dict(estimate=mean_defined(subjects),valid_subjects=int(np.isfinite(vector).sum()),**interval(means))
    def collect(records):
        arrays = {k:np.array([p[k] for p in records]) for k in records[0]}
        return ({k:v.sum() for k,v in arrays.items()},
                {k:v[draws].sum(1) for k,v in arrays.items()})
    totals, sampled = collect(person_counts)
    cases, cases_sampled = collect(person_cases)
    for key in cases:
        contribution[f'impact/{key}'] = [row[key] for row in person_cases]
    counts_all_seeds = {field:sum(night['counts'][field] for ns in people.values() for night in ns.values())
                        for field in totals}
    global_scores = f1_scores(counts_all_seeds)
    score_repeats = f1_scores(sampled)
    impact = repair_gains(totals,cases)
    impact_repeats = repair_gains(sampled,cases_sampled)
    f1_impact = {k:dict(delta=float(v),mean_seed_total=float(cases[k]),**interval(impact_repeats[k])) for k,v in impact.items()}
    hypothesis_contrasts = {}
    for key in ('q_wrong','b_blocks_correct_q','hyp_loses_to_pause'):
        difference = impact_repeats['hyp_misses']-impact_repeats[key]
        hypothesis_contrasts['hyp_misses_minus_'+key] = dict(delta=float(impact['hyp_misses']-impact[key]),**interval(difference))
    all_events = [e for nights in people.values() for night in nights.values() for e in night['events']]
    q_pooled = {}
    for unit,matrices in q_nights.items():
        matrices = np.asarray(matrices)
        sampled_q = matrices[draws].sum(1)
        den = sampled_q.sum(1)+sampled_q.sum(2)
        f1 = np.divide(2*sampled_q.diagonal(axis1=1,axis2=2),den,
                       out=np.zeros_like(den,dtype=float),where=den>0).mean(1)
        f1[sampled_q.sum((1,2))==0] = np.nan
        q_pooled[unit] = dict(**matrix_metrics(matrices.sum(0)),bootstrap_macro_f1=interval(f1))
    return dict(subject_equal_bootstrap=stats,subject_seed_means=averaged,
                person_count_seed_means=dict(zip(names,person_counts,strict=True)),
                person_error_seed_means=dict(zip(names,person_cases,strict=True)),
                event_seed_observations=event_stats(all_events),
                cohort_count_seed_means=totals,
                q_pooled_seed_mean_confusion=q_pooled,
                production={k:dict(estimate=float(v),**interval(score_repeats[k])) for k,v in global_scores.items()},
                score_impact=f1_impact,score_impact_contrasts=hypothesis_contrasts,
                concentration={k:top_contribution(np.asarray(v),names) for k,v in contribution.items()},
                bootstrap=dict(samples=BOOT,seed=SEED,unit='受试者',stratified=False,people=names,
                               sample_indices_sha256=audit.digest(draws.tobytes())))


def read_archives(prior,causal,limit,progress):
    roster = repro.roster()
    labels = audit.reference_labels(roster)
    people = {person:{} for person in sorted(labels)}
    progress['missing_runs'] = [f'{fold}/{seed}' for fold in roster for seed in track.SEEDS]
    identity = {}
    reference = causal['identity']['T03']
    assert prior['T03']==reference['t03']==audit.VERSION
    expected = prior['archive']['archives']['1/4121']
    assert reference['arrays']==expected['arrays_sha256'] and reference['states']==expected['states_sha256']
    assert causal['status']=='completed' and len(causal['causal_max_abs'])==5 and max(causal['causal_max_abs'])<TOLERANCE
    for fold,roles in roster.items():
        for seed in track.SEEDS:
            limit()
            key = f'{fold}/{seed}'
            progress['current_run'] = key
            receipt,states,arrays,source = repro.t03_run(fold,seed)
            assert receipt['subjects']==roles and source==prior['archive']['source']
            identity[key] = dict(states_sha256=audit.digest(states),arrays_sha256=audit.digest(arrays),
                                 receipt_hash=audit.digest(json.dumps(receipt,sort_keys=True).encode()))
            assert identity[key]==prior['archive']['archives'][key]
            saved_counts = {row['person']:row for row in receipt['results']}
            assert saved_counts.keys()==set(roles['select'])
            with np.load(io.BytesIO(arrays)) as archive:
                assert {k for k in archive.files if k not in ('train_index','select_index') and not k.endswith('_second')}==set(roles['select'])
                for person in roles['select']:
                    progress['current_person'] = person
                    target,clock = labels[person]
                    y,p = target[:,-1],archive[person]
                    audit.check_predictions(p,archive[person+'_second'],clock)
                    before = prior['archive']['jobs'][key]['validation'][person]
                    report = one_night(y,p,clock,before,saved_counts[person])
                    assert str(seed) not in people[person]
                    people[person][str(seed)] = dict(fold=fold,seed=seed,**report)
            progress['missing_runs'].remove(key)
            progress['validated_runs'].append(key)
            print(f'内折 {fold}、{seed} 报警、匹配及分型已通过逐夜核验',flush=True)
    assert len(people)==20 and all(set(ns)=={str(s) for s in track.SEEDS} for ns in people.values())
    return people,identity,source


def save(path,data):
    body = json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
    repro.atomic(path,lambda f:f.write(body))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--started-at',type=float,default=monotonic())
    args = parser.parse_args()
    def limit():
        if monotonic()-args.started_at>=600:
            raise TimeoutError('达到 600 秒软限')
    limit()
    def hard_timeout(_signum,_frame):
        raise TimeoutError('达到 900 秒硬限')
    signal.signal(signal.SIGALRM,hard_timeout)
    signal.alarm(max(1,int(900-(monotonic()-args.started_at))))
    progress = dict(validated_runs=[])
    begin = monotonic()
    try:
        r0_path, r1_path = (repro.OUTPUT/f'BreathLLM-R{i}结果.json' for i in (0,1))
        prior,causal = (json.loads(path.read_text()) for path in (r0_path,r1_path))
        people,archives,sources = read_archives(prior,causal,limit,progress)
        audit_seconds = monotonic()-begin
        result = analysis(people,limit)
        result_f1 = result['production']
        old_counts = [r['counts'] for job in prior['archive']['jobs'].values() for r in job['validation'].values()]
        old = f1_scores({k:sum(row[k] for row in old_counts) for k in ('seconds','truth','predicted','detected','typed')})
        for key,value in old.items():
            assert result_f1[key]['estimate']==value
        invalid = {k:sum(n['invalid'][k] for nights in people.values() for n in nights.values()) for k in ('b','q','small_b_range_excursions')}
        max_identities = {k:max(n['identities'][k] for nights in people.values() for n in nights.values())
                          for k in ('normal','pause','hyp','q_mass','event_mass')}
        causality = dict(R1_sha256=track.hash_file(r1_path),fitted_window_future_perturbation=causal['causal_max_abs'],
                         archive='1/4121',all_archive_original_causal_test=True,
                         inference_provenance=['FoldData.windows 只切取当前 480 点',
                                               '波形降采样使用单向 sosfilt，缩放统计来自拟合侧',
                                               'B07 score 先完成末位置推理，再读取标签评分',
                                               '每份归档生成前在五个监督位置执行未来扰动测试'])
        report = dict(status='completed',T03=audit.VERSION,protocol=b07.PROTOCOL,archives=archives,source=sources,
                      source_sha256=track.hash_file(Path(__file__)),R0_sha256=track.hash_file(r0_path),causality=causality,
                      audit=dict(people=len(people),runs=len(archives),nights=sum(map(len,people.values())),
                                 max_identities=max_identities,invalid=invalid,reference_production=old,
                                 fp64_a_threshold_disagreements=sum(n['fp64_a_threshold_disagreements'] for ns in people.values() for n in ns.values()),
                                 max_alarm_score_difference=max(n['max_alarm_score_difference'] for ns in people.values() for n in ns.values())),
                      subjects=people,statistics=result,phases=dict(archive=audit_seconds,statistics=monotonic()-begin-audit_seconds),
                      seconds=monotonic()-args.started_at)
        save(OUT,report)
        print(f'R2 完成，累计 {report["seconds"]:.2f}s',flush=True)
    except Exception as error:
        save(FAILURE,dict(status='partial',stage=progress,error=repr(error),seconds=monotonic()-args.started_at))
        raise
    finally:
        signal.alarm(0)


if __name__=='__main__':
    main()
