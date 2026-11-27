"""冻结 T03 基座、数据覆盖及概率瓶颈的只读审计。"""

import argparse
import io
import json
import signal
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import 发布复现 as repro
from src.experiment_data import waveform

audit, track, b07 = repro.audit, repro.track, repro.b07
OUT = repro.OUTPUT / 'BreathLLM-R0结果.json'
CLASSES = tuple(b07.CLASS_NAMES)
ABNORMAL = CLASSES[1:]
BINS = np.linspace(0,1,11)
QUANTILES = (.1,.25,.5,.75,.9)


def quantiles(x):
    return np.quantile(x,QUANTILES).tolist() if len(x) else None


def equal_quantiles(points):
    if not points:
        return dict(values=None,subjects=0)
    values = np.concatenate(points)
    weights = np.concatenate([np.full(len(v),1/len(v)) for v in points])
    order = np.argsort(values)
    idx = np.searchsorted(weights[order].cumsum()/len(points),QUANTILES)
    return dict(values=values[order[idx]].tolist(),subjects=len(points))


def counts(y):
    return dict(zip(CLASSES,np.bincount(y[y>=0],minlength=5).tolist()))


def coverage_summary(people, field):
    result = {}
    for name in ABNORMAL:
        sizes = {p:row[field][name] for p,row in people.items() if row[field][name]>0}
        vector = np.array(list(sizes.values()),dtype=float)
        total = float(vector.sum())
        result[name] = dict(events=total,subjects=len(sizes),n_eff=float(total**2/(vector@vector)) if total else 0.,
                            per_person=sizes,top_person=max(sizes,key=sizes.get) if total else None,
                            top_share=float(vector.max()/total) if total else None)
    return result


def event_counts(events, mask):
    return counts(events[mask,2].astype(int))


def clock_events(labels):
    result = {}
    for person,(target,clock) in labels.items():
        extended = np.r_[target[0,:-1],target[:,-1]]
        events = b07.runs(np.maximum(extended,0))
        events[:,:2] += clock[0]-4
        by_second = np.full(len(extended),-1,dtype=int)
        for i,(left,right,_) in enumerate(events):
            by_second[left-clock[0]+4:right-clock[0]+4] = i
        result[person] = events,by_second
    return result


def fit_coverage(schedule,roles,labels,events_index):
    rows = {}
    for i,p in enumerate(roles['fit']):
        target,clock = labels[p]
        selected = schedule[schedule[:,0]==i,1]
        seconds = (clock[selected,None]-np.arange(4,-1,-1)).reshape(-1)
        label = target[selected].reshape(-1)
        events,lookup = events_index[p]
        ids = lookup[seconds-clock[0]+4]
        np.testing.assert_array_equal(ids>=0,label>0)
        covered = np.unique(ids[ids>=0])
        _,unique = np.unique(seconds,return_index=True)
        duration = {name:(events[covered,1]-events[covered,0])[events[covered,2]==cls].tolist()
                    for cls,name in enumerate(ABNORMAL,1)}
        rows[p] = dict(windows=len(selected),supervised=counts(label),unique_supervised=counts(label[unique]),
                       invalid=int((label<0).sum()),covered=event_counts(events,np.isin(np.arange(len(events)),covered)),
                       covered_duration_seconds=duration,covered_event_spans=events[covered].tolist())
    return rows


def branches(p):
    a = 1.-p[:,0]
    ab = p[:,1:4].sum(1)
    b = np.divide(ab,a,out=np.full(len(p),np.nan),where=a>0)
    q = np.divide(p[:,1:4],ab[:,None],out=np.full((len(p),3),np.nan),where=ab[:,None]>0)
    return a,b,q


def evidence(y,p):
    a,b,q = branches(p)
    apnea = (y>0)&(y<4)
    valid = apnea&np.isfinite(b)&np.isfinite(q).all(1)
    q_y = q[valid,y[valid]-1]
    q_max = q[valid].max(1)
    competition = b[valid]*q_max-(1-b[valid])
    true_margin = b[valid]*q_y-(1-b[valid])
    q_correct = q[valid].argmax(1)+1==y[valid]
    other = p[:,1:].max(1)
    flags = dict(any_blocked=competition<=0,q_error_after_b=(competition>0)&~q_correct,
                 correct_q_blocked=q_correct&(true_margin<=0),q_error=~q_correct,
                 b_below_half=b[valid]<.5,b_below_true_cut=b[valid]<1/(1+q_y),
                 normal_wins=p[valid,0]>other[valid],
                 abnormal_below_alarm_threshold=p[valid,1:].astype(np.float32).sum(1)<.5,
                 final_correct=p[valid].argmax(1)==y[valid],
                 b_block_without_normal=(competition<=0)&(p[valid,0]<=other[valid]))
    return dict(labels=y,b=b,seconds=dict(y=y[valid],m_any=competition,m_true=true_margin,flags=flags),
                invalid=dict(a_nonpositive=int((a<=0).sum()),zero_pause_mass=int((~np.isfinite(q).all(1)).sum()),
                             b_undefined_abnormal=int(((y>0)&~np.isfinite(b)).sum()),
                             apnea_margin_undefined=int((apnea&~valid).sum())),
                normal_wins_on_anomaly={name:int(((y==cls)&(p[:,0]>other)).sum()) for cls,name in enumerate(ABNORMAL,1)})


def event_evidence(y, p, positive, matches):
    # Same cumulative summation and dtype as the production scorer, channel by channel.
    mass = np.vstack((np.zeros(5),np.cumsum(p*positive[:,None],axis=0)))
    events = []
    for row in matches:
        left,right,label,guess,dl,dr = row
        left,right,label = int(left),int(right),int(label)
        entry = dict(reference=[left,right],label=label,duration=right-left,matched=bool(np.isfinite(guess)))
        if entry['matched']:
            begin,end = int(left+dl),int(right+dr)
            sums = mass[end]-mass[begin]
            assert sums[1:].argmax()+1==int(guess)
            entry.update(alarm=[begin,end],prediction=int(guess),scores=sums.tolist(),positive_seconds=int(positive[begin:end].sum()))
        events.append(entry)
    matched_apnea = [r for r in events if r['matched'] and r['label']<4]
    score = np.array([r['scores'] for r in matched_apnea]).reshape(-1,5)
    truth = np.array([r['label'] for r in matched_apnea],dtype=int)
    any_margin = score[:,1:4].max(1)-score[:,4]
    true_margin = score[np.arange(len(truth)),truth]-score[:,4]
    q_correct = score[:,1:4].argmax(1)+1==truth
    checks = dict(any_blocked=any_margin<=0,q_error_after_b=(any_margin>0)&~q_correct,
                  correct_q_blocked=q_correct&(true_margin<=0),q_error=~q_correct,
                  normal_wins=score[:,0]>score[:,1:].max(1),final_correct=score[:,1:].argmax(1)+1==truth,
                  b_block_without_normal=(any_margin<=0)&(score[:,0]<=score[:,1:].max(1)))
    return events,dict(y=truth,m_any=any_margin,m_true=true_margin,flags=checks)


def margins(parts):
    y = np.concatenate([x['y'] for x in parts])
    joined = {k:np.concatenate([x[k] for x in parts]) for k in ('m_any','m_true')}
    indicators = {k:np.concatenate([x['flags'][k] for x in parts]) for k in parts[0]['flags']}
    result = {}
    for i,name in enumerate(('全部暂停',*CLASSES[1:4])):
        use = y>0 if i==0 else y==i
        n = int(use.sum())
        result[name] = dict(n=n,quantiles={k:quantiles(x[use]) for k,x in joined.items()},
                            rates={k:dict(count=int(x[use].sum()),rate=float(x[use].mean()) if n else None)
                                   for k,x in indicators.items()})
    return result


def probability_bins(y,b):
    # A sub-ULP discrepancy from the archived float32 normal probability can put b just above 1.
    idx = np.minimum(np.searchsorted(BINS,b,side='right')-1,9)
    support = np.bincount(idx,minlength=10)
    means = np.bincount(idx,weights=b,minlength=10)
    outcomes = np.bincount(idx,weights=y,minlength=10)
    return support,means,outcomes


def calibration(group, include_equal=True):
    individuals = {}
    packs = []
    for person,nights in group.items():
        yy = np.concatenate([r['labels'] for r in nights])
        pp = np.concatenate([r['b'] for r in nights])
        selected = (yy>0)&np.isfinite(pp)
        assert not (pp[selected]<0).any()
        packs.append((person,yy[selected],pp[selected]))
    y = np.concatenate([r[1] for r in packs])
    score = np.concatenate([r[2] for r in packs])
    truth = y<4
    def metrics(label,p):
        if not len(label):
            return dict(AUROC=None,AUPRC=None,Brier=None,prevalence=None,mean_b=None)
        return dict(AUROC=float(roc_auc_score(label,p)) if label.any() and not label.all() else None,
                    AUPRC=float(average_precision_score(label,p)) if label.any() else None,
                    Brier=float(np.square(p-label).mean()),prevalence=float(label.mean()),mean_b=float(p.mean()))
    def table(support,sums,observed):
        return [dict(interval=[float(BINS[i]),float(BINS[i+1])],n=int(n),mean_prediction=float(p/n) if n else None,
                     fraction_apnea=float(t/n) if n else None) for i,(n,p,t) in enumerate(zip(support,sums,observed,strict=True))]
    support,sums,observed = probability_bins(truth,score)
    pooled = dict(metrics=metrics(truth,score),bins=table(support,sums,observed),
                  quantiles={name:quantiles(score[y==i]) for i,name in enumerate(ABNORMAL,1)},
                  out_of_unit_interval=int(((score<0)|(score>1)).sum()),maximum_b=float(score.max()),
                  Brier_prevalence_reference=float(truth.mean()*(1-truth.mean())),
                  ECE=float(abs(sums-observed).sum()/len(truth)))
    bins = []
    for p,yy,b in packs:
        stat = probability_bins(yy<4,b)
        bins.append(stat)
        individuals[p] = dict(metrics=metrics(yy<4,b),bins=table(*stat),quantiles={name:quantiles(b[yy==i])
                                                                                  for i,name in enumerate(ABNORMAL,1)})
    if not include_equal:
        return dict(pooled=pooled,subjects=individuals)
    equal_metrics = {}
    for field in pooled['metrics']:
        vals = [v['metrics'][field] for v in individuals.values() if v['metrics'][field] is not None]
        equal_metrics[field] = dict(mean=float(np.mean(vals)) if vals else None,subjects=len(vals))
    equal_bins = []
    for i in range(10):
        selected = [row['bins'][i] for row in individuals.values() if row['bins'][i]['n']]
        equal_bins.append(dict(interval=pooled['bins'][i]['interval'],subjects=len(selected),n=sum(x['n'] for x in selected),
                               mean_prediction=float(np.mean([d['mean_prediction'] for d in selected])) if selected else None,
                               fraction_apnea=float(np.mean([d['fraction_apnea'] for d in selected])) if selected else None))
    quantile_by_class = {name:equal_quantiles([b[yy==i] for _,yy,b in packs if (yy==i).any()])
                         for i,name in enumerate(ABNORMAL,1)}
    subject_ece = [float(abs(s-t).sum()/n.sum()) for n,s,t in bins if n.sum()]
    return dict(pooled=pooled,equal_subject=dict(metrics=equal_metrics,bins=equal_bins,quantiles=quantile_by_class,
                                                 ECE=float(np.mean(subject_ece))),subjects=individuals)


def by_scope(group):
    nights = [n for items in group.values() for n in items]
    secondary = {name:sum(n['invalid'][name] for n in nights) for name in nights[0]['invalid']}
    summaries = {}
    for unit,name in (('seconds','seconds'),('event_evidence','events')):
        pooled = margins([n[unit] for n in nights])
        per_person = [margins([n[unit] for n in observations]) for observations in group.values()]
        for i,cls in enumerate(('全部暂停',*CLASSES[1:4])):
            y = [np.concatenate([n[unit]['y'] for n in observations]) for observations in group.values()]
            use = [(v==i if i else v>0) for v in y]
            rates = {}
            for key in pooled[cls]['rates']:
                values = [p[cls]['rates'][key]['rate'] for p in per_person if p[cls]['n']]
                rates[key] = dict(mean=float(np.mean(values)) if values else None,subjects=len(values))
            def dist(key):
                data = [np.concatenate([n[unit][key] for n in obs])[mask]
                        for obs,mask in zip(group.values(),use,strict=True) if mask.any()]
                return equal_quantiles(data)
            pooled[cls]['equal_subject'] = dict(rates=rates,quantiles={k:dist(k) for k in ('m_any','m_true')})
        summaries[name] = pooled
    return dict(**summaries,
                invalid=secondary,normal_wins_on_anomaly={k:sum(n['normal_wins_on_anomaly'][k] for n in nights) for k in ABNORMAL},
                calibration=calibration(group))


def audit_all(limit):
    started = monotonic()
    roster = repro.roster()
    labels = audit.reference_labels(roster)
    indexed = clock_events(labels)
    datasets, jobs, records, archive_source = {},{},[],{}
    dev = {p:dict(truth=counts(events[:,2]),duration_seconds={name:(events[events[:,2]==i,1]-events[events[:,2]==i,0]).tolist()
                                                          for i,name in enumerate(ABNORMAL,1)},spans=events.tolist())
           for p,(events,_) in indexed.items()}
    for fold,roles in roster.items():
        for seed in track.SEEDS:
            limit()
            receipt,states,arrays,source = repro.t03_run(fold,seed)
            assert receipt['subjects']==roles
            run_key = f'{fold}/{seed}'
            archive_source[run_key] = dict(states_sha256=audit.digest(states),arrays_sha256=audit.digest(arrays),
                                           receipt_hash=audit.digest(json.dumps(receipt,sort_keys=True).encode()))
            with np.load(io.BytesIO(arrays)) as archived:
                train,select = archived['train_index'],archived['select_index']
                clockset = SimpleNamespace(roles=roles,labels={p:r[0] for p,r in labels.items()})
                np.testing.assert_array_equal(train,track.sample_clock(clockset,'fit',track.STEPS*track.BATCH,seed,True))
                np.testing.assert_array_equal(select,track.sample_clock(clockset,'select',track.SELECT_SIZE,74121,False))
                end = receipt['fit']['curve'][-1]['step']
                best = receipt['fit']['best_step']
                fit = fit_coverage(train[:end*track.BATCH],roles,labels,indexed)
                best_fit = fit if best==end else fit_coverage(train[:best*track.BATCH],roles,labels,indexed)
                validations = {}
                for p in roles['select']:
                    limit()
                    target,clock = labels[p]
                    p5 = archived[p]
                    audit.check_predictions(p5,archived[p+'_second'],clock)
                    y = target[:,-1]
                    score,positive,match = b07.event_score(y,p5)
                    audit.check_matches(y,positive,match)
                    saved = next(item for item in receipt['results'] if item['person']==p)
                    assert all(v==saved[k] for k,v in score.items())
                    events,measures = event_evidence(y,p5,positive,match)
                    events_clock = [dict(e,reference=[v+int(clock[0]) for v in e['reference']],
                                         **({'alarm':[v+int(clock[0]) for v in e['alarm']]} if e['matched'] else {})) for e in events]
                    valid = np.isfinite(match[:,3])
                    typed = valid&(match[:,2]==match[:,3])
                    match_count = event_counts(match,valid)
                    truth_count = event_counts(match,np.ones(len(match),dtype=bool))
                    # Coverage starts at the first released prediction, while fit coverage includes the preceding four labels.
                    assert truth_count==counts(b07.runs(np.maximum(y,0))[:,2])
                    validations[p] = dict(matched=match_count,correct=event_counts(match,typed),truth=truth_count,
                                          event_cm=score['event_cm'],counts=score,events=events_clock,
                                          matched_duration_seconds={name:[e['duration'] for e in events if e['matched'] and e['label']==i]
                                                                    for i,name in enumerate(ABNORMAL,1)})
                    records.append((fold,seed,p,dict(**evidence(y,p5.astype(np.float64)),event_evidence=measures)))
            jobs[run_key]=dict(fit=fit,fit_at_best=best_fit,validation=validations,ran_steps=end,best_step=best,
                               fit_index_hash=audit.digest(train.tobytes()),select_index_hash=audit.digest(select.tobytes()),
                               concentration={k:coverage_summary(fit if k=='covered' else validations,k)
                                              for k in ('covered','truth','matched','correct')},
                               fit_windows_planned=len(train))
            datasets[run_key] = {p:by_scope({p:[v]}) for f,s,p,v in records if f==fold and s==seed}
            print(f'T03 内折 {fold}、{seed} 归档和训练覆盖已核对',flush=True)
    people = sorted(dev)
    complete = {p:[x for _,_,person,x in records if person==p] for p in people}
    assert len(people)==20 and all(len(data)==3 for data in complete.values())
    aggregate = by_scope(complete)
    per_seed, per_fold = {},{}
    for seed in track.SEEDS:
        per_seed[seed] = by_scope({p:[x for _,s,person,x in records if person==p and s==seed] for p in people})
    for fold,roles in roster.items():
        per_fold[fold] = by_scope({p:complete[p] for p in roles['select']})
    for p in people:
        joined = [job['validation'][p] for job in jobs.values() if p in job['validation']]
        dev[p]['scored_truth'] = joined[0]['truth']
        for field in ('matched','correct'):
            dev[p][field] = {c:sum(x[field][c] for x in joined)/len(joined) for c in ABNORMAL}
        covered = np.array(sorted({tuple(e) for row in jobs.values() if p in row['fit'] for e in row['fit'][p]['covered_event_spans']})).reshape(-1,3)
        dev[p]['covered'] = event_counts(covered,np.ones(len(covered),dtype=bool))
    for seed in track.SEEDS:
        cohort = {p:row for k,job in jobs.items() if k.endswith(f'/{seed}') for p,row in job['validation'].items()}
        assert len(cohort)==20
        per_seed[seed]['concentration'] = {field:coverage_summary(cohort,field) for field in ('truth','matched','correct')}
    return dict(archives=archive_source,source=source,protocol=b07.PROTOCOL,development=dev,
                concentration={k:coverage_summary(dev,k) for k in ('truth','scored_truth','covered','matched','correct')},
                jobs=jobs,by_seed=per_seed,by_fold=per_fold,per_run_subject=datasets,aggregate=aggregate,
                seconds=monotonic()-started)


def tensor_layout(x):
    if x is None:
        return None
    return dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),device=str(x.device))


def tensor_fingerprint(x):
    if x is None:
        return None
    return dict(**tensor_layout(x),sha256=audit.digest(x.detach().contiguous().view(-1).cpu().float().numpy().tobytes()),
                values=x.cpu().tolist() if x.numel()<=32 else None)


def tracer(ref):
    import torch
    from contextlib import ExitStack

    model = ref.model
    base = model.base
    backbone = base.backbone
    gemma = backbone.llm.get_base_model().model

    @torch.no_grad()
    def capture(x, select):
        values, layout, masks, positions = {}, {}, {}, {}
        n = len(x)
        take = torch.as_tensor(select,device=x.device)
        def observe(name,out,channels=False):
            if not isinstance(out,torch.Tensor):
                out = out.last_hidden_state if hasattr(out,'last_hidden_state') else out[0]
            layout[name] = tensor_layout(out)
            if channels:
                out = out.reshape(n,4,*out.shape[1:])
            values[name] = out[take].detach().float().cpu().numpy()
        def output_hook(name,channels=False):
            return lambda _m,_a,out:observe(name,out,channels)
        def inputs_hook(_m,_a,kw):
            observe('Gemma input',kw['inputs_embeds'])
            masks['input']=tensor_fingerprint(kw.get('attention_mask'))
            positions['input']=tensor_fingerprint(kw.get('position_ids'))
            assert not kw.get('use_cache')
        def decoder_inputs(name,_m,_a,kw):
            mask = kw.get('attention_mask')
            pos = kw.get('position_ids')
            masks[name] = tensor_fingerprint(mask)
            positions[name] = tensor_fingerprint(pos)
        with ExitStack() as stack:
            observe('T03 input',x)
            enc = backbone.encoder.encoder.layers[0]
            stack.callback(enc.register_forward_hook(
                lambda _m,_a,out:observe('encoder '+str(out.shape[1]),out,True)).remove)
            stack.callback(backbone.projector.register_forward_hook(output_hook('time projection')).remove)
            stack.callback(base.tail_projection.register_forward_hook(output_hook('tail projection')).remove)
            stack.callback(gemma.register_forward_pre_hook(inputs_hook,with_kwargs=True).remove)
            for i,layer in enumerate(gemma.layers):
                name = f'Gemma layer {i+1}'
                stack.callback(layer.register_forward_pre_hook(
                    lambda m,a,kw,name=name:decoder_inputs(name,m,a,kw),with_kwargs=True).remove)
                stack.callback(layer.register_forward_hook(output_hook(name)).remove)
            for name in ('q_proj','v_proj'):
                module = getattr(gemma.layers[0].self_attn,name)
                stack.callback(module.register_forward_hook(output_hook(f'Gemma layer 1 {name}')).remove)
            for name in ('event','apnea','subtype'):
                stack.callback(getattr(base.physiology,name).register_forward_hook(output_hook('physiology '+name)).remove)
            out = model(x)
            for name,z in zip(('a logit','b logit','q logits'),out.conditional_logits,strict=True):
                observe(name,z)
            observe('log probability',out.log_prob)
            observe('probability',out.log_prob.exp())
        return values,dict(tensors=layout,attention_mask=masks,position_ids=positions,padding=0)
    return capture


def compare_layers(control,candidate):
    values = {}
    for k,y in candidate.items():
        y = y.astype(np.float64)
        x = control[k].astype(np.float64)
        assert x.shape==y.shape
        d = y-x
        norm = np.linalg.norm(x)
        values[k] = dict(max_abs=float(np.max(np.abs(d))),rmse=float(np.sqrt(np.square(d).mean())),
                         relative_l2=float(np.linalg.norm(d)/norm) if norm else None)
    p,q = control['probability'],candidate['probability']
    decision = p.argmax(-1)!=q.argmax(-1)
    alarm = (p[...,1:].sum(-1)>=.5)!=(q[...,1:].sum(-1)>=.5)
    return dict(layers=values,decision_flips=int(decision.sum()),latest_decision_flips=int(decision[:,-1].sum()),
                alarm_flips=int(alarm.sum()),latest_alarm_flips=int(alarm[:,-1].sum()))


def supplement(limit):
    import torch
    from contextlib import nullcontext
    from torch.nn.attention import SDPBackend, sdpa_kernel

    started = monotonic()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    ref,identity,replay = repro.reference()
    for parameter in ref.params.values():
        parameter.requires_grad_(True)
    ref.model.eval()
    schedule = ref.train[:48]
    x,_ = ref.batch('fit',schedule)
    raw = np.empty((16,4,480),dtype=np.float32)
    raw_source = {}
    for i in np.unique(schedule[:16,0]):
        person = ref.data.roles['fit'][i]
        path = repro.DATA/f'{person}.rec'
        source = waveform(path,track.PSG)
        raw_source[person]=track.hash_file(path)
        use = np.flatnonzero(schedule[:16,0]==i)
        ends = ref.data.ends[person][schedule[use,1]]
        raw[use] = source[:,(ends[:,None]-60)*8+np.arange(480)].transpose(1,0,2)
    expected = np.nan_to_num((raw-ref.data.mean[None,:,None])/ref.data.scale[None,:,None]).astype(np.float32)
    np.testing.assert_array_equal(expected,x[:16].cpu().numpy())
    backbone = ref.model.base.backbone
    trace = tracer(ref)
    results = {}
    original = None
    for precision in ('native','fp32_math'):
        limit()
        if precision=='fp32_math':
            backbone.llm.float()
            backbone.fixed_prefix = backbone.fixed_prefix.float()
            torch.set_float32_matmul_precision('highest')
            torch.backends.cudnn.allow_tf32=False
            torch.backends.mha.set_fastpath_enabled(False)
        config = dict(gemma_attention=backbone.llm.get_base_model().model.config._attn_implementation,
                      torch=str(torch.__version__),cuda=torch.version.cuda,device=torch.cuda.get_device_name(ref.device) if ref.device.type=='cuda' else str(ref.device),
                      cudnn=torch.backends.cudnn.version(),matmul=torch.get_float32_matmul_precision(),
                      cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                      mha_fastpath=torch.backends.mha.get_fastpath_enabled(),
                      dropout_in_eval=all(not m.training for m in ref.model.modules()),
                      dispatch=dict(flash=torch.backends.cuda.flash_sdp_enabled(),math=torch.backends.cuda.math_sdp_enabled(),
                                    efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),cudnn=torch.backends.cuda.cudnn_sdp_enabled()))
        context = sdpa_kernel(SDPBackend.MATH) if precision=='fp32_math' else nullcontext()
        with context:
            control,control_layout = trace(x[:32],np.arange(16))
            cases = {'repeat':(x[:32],np.arange(16)), 'batch_16':(x[:16],np.arange(16)),
                     'reorder':(x[:32].flip(0),np.arange(31,15,-1)),
                     'replace_companions':(torch.cat((x[:16],x[32:48])),np.arange(16))}
            comparisons = {}
            for name,(batch,indices) in cases.items():
                limit()
                data,layout = trace(batch,indices)
                np.testing.assert_array_equal(data['T03 input'],control['T03 input'])
                comparisons[name]=dict(**compare_layers(control,data),layouts=[layout])
            for size in (1,4):
                blocks,layouts=[],[]
                for section in x[:16].split(size):
                    limit()
                    part,layout=trace(section,np.arange(len(section)))
                    blocks.append(part)
                    layouts.append(layout)
                data={k:np.concatenate([part[k] for part in blocks],axis=0) for k in control}
                np.testing.assert_array_equal(data['T03 input'],control['T03 input'])
                comparisons[f'batch_{size}']=dict(**compare_layers(control,data),layouts=layouts)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                trace(x[:4],np.arange(4))
            config['observed_attention_ops'] = sorted({e.key for e in prof.key_averages() if 'scaled_dot_product' in e.key})
        if original is None:
            original = control
        else:
            results['fp32_vs_native_same_batch']=compare_layers(original,control)
        config['allowed_gemma_backend'] = 'MATH' if precision=='fp32_math' else 'original SDPA dispatcher'
        maximum = max(d['layers']['probability']['max_abs'] for d in comparisons.values())
        maximum_log = max(d['layers']['log probability']['max_abs'] for d in comparisons.values())
        decision_flips = sum(d['decision_flips'] for d in comparisons.values())
        alarm_flips = sum(d['alarm_flips'] for d in comparisons.values())
        results[precision]=dict(config=config,control_layout=control_layout,comparisons=comparisons,
                                max_probability=maximum,max_log_probability=maximum_log,decision_flips=decision_flips,
                                alarm_flips=alarm_flips,engineering_pass=maximum<=1e-4 and maximum_log<=1e-3 and not (decision_flips or alarm_flips))
    anchors = [(ref.data.roles['fit'][i],int(ref.data.ends[ref.data.roles['fit'][i]][j])-1) for i,j in schedule[:16]]
    return dict(methods=results,windows=anchors,all_indices=schedule.tolist(),
                raw_psg=dict(shape=list(raw.shape),stride=list(raw.strides),dtype=str(raw.dtype),sha256=audit.digest(raw.tobytes()),
                             sources=raw_source),cache_replay=replay,seconds=monotonic()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--started-at',type=float,default=monotonic())
    parser.add_argument('--statistics-only',action='store_true')
    args = parser.parse_args()
    elapsed = monotonic()-args.started_at
    if not 0<=elapsed<600:
        parser.error('批次起点须在有效软限内')
    signal.alarm(int(900-elapsed))
    def limit():
        if monotonic()-args.started_at>=600:
            raise TimeoutError('达到整批软限')
    try:
        archive = audit_all(limit)
        history_path = repro.PROJECT/'config/历史汇总.json'
        history = dict(P2_amplitude=json.loads(history_path.read_text())['P2'],source=history_path.name,
                       source_sha256=track.hash_file(history_path))
        batch = supplement(limit) if not args.statistics_only else None
        result = dict(status='completed',T03=audit.VERSION,archive=archive,history=history,batch=batch,
                      source_sha256=track.hash_file(Path(__file__)),seconds=monotonic()-args.started_at)
        body = json.dumps(result,ensure_ascii=False,allow_nan=False).encode()
        repro.atomic(OUT,lambda f:f.write(body))
        print('R0 核对完成',result['seconds'],flush=True)
    finally:
        signal.alarm(0)


if __name__=='__main__':
    main()
