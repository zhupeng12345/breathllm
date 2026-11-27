"""T03 暂停排序与校准的预设离线探针及 FP32 工程复核。"""

import argparse
import importlib
import io
import json
import signal
import sqlite3
from contextlib import closing
from pathlib import Path
from time import monotonic, perf_counter

import numpy as np
import torch
from scipy.special import expit, softmax
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F

r0 = importlib.import_module('BreathLLM-R0')
import 发布复现 as repro

audit, track, b07 = repro.audit, repro.track, repro.b07
OUT = repro.OUTPUT / 'BreathLLM-R1结果.json'
RECORDS = repro.OUTPUT / 'R1探针.sqlite'
VALUE_WIDTH, AMPLITUDE_WIDTH = 27, 24
ARMS = ('R00','R01','R02','R03','R04','R05')
SEED, STEPS, BOOT = 4121, 128, 10000
RANK_WEIGHT, SCALE_FLOOR = .25, 1e-6


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            self.layers = nn.Sequential(nn.Linear(width,24), nn.GELU(), nn.Linear(24,1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, h, b):
        return b + self.layers(h).squeeze(-1)


class Calibration(nn.Module):
    def __init__(self, affine):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.theta = nn.Parameter(torch.zeros(())) if affine else None

    def forward(self, h, b):
        return (self.theta.exp()*b if self.theta is not None else b) + self.bias


def rank_loss(z, y, people, within):
    positive = (y > 0) & (y < 4)
    negative = y == 4
    left, right = z[positive], z[negative]
    if not left.numel() or not right.numel():
        return z.sum()*0, 0, 0
    pi, ni = people[positive], people[negative]
    pair = F.softplus(right[None,:] - left[:,None])
    if not within:
        return pair.mean(), pair.numel(), int(torch.unique(torch.cat((pi,ni))).numel())
    loss, count = [], 0
    for person in pi.unique():
        local = pair[(pi==person)[:,None] & (ni==person)[None,:]]
        if local.numel():
            loss.append(local.mean())
            count += local.numel()
    return (torch.stack(loss).mean() if loss else z.sum()*0), count, len(loss)


def train_loss(model, arm, h, logits, y, people):
    a,b,q = logits
    z = model(h,b)
    lp = track.five_class_log_probs(a,z,q)
    nll = F.nll_loss(lp.flatten(0,1),y.flatten(),ignore_index=-1)
    valid, abnormal = int((y>=0).sum()), int((y>0).sum())
    rank, pairs, participants = (rank_loss(z,y,people,arm=='R05')
                                 if arm in ('R04','R05') else (z.new_zeros(()),0,0))
    loss = nll+RANK_WEIGHT*abnormal/max(valid,1)*rank
    return loss, dict(nll=nll.item(),rank_loss=rank.item(),pairs=pairs,participants=participants,
                      valid=valid,abnormal=abnormal,total=loss.item())


class HiddenInput:
    """沿用 A01 的固定投影、24 个启用数值槽及对应掩码。"""
    def __init__(self, hidden, labels):
        valid = hidden[labels>=0].float()
        self.mean = valid.mean(0)
        self.std = valid.std(0,unbiased=False).clamp_min(SCALE_FLOOR)
        rng = torch.Generator().manual_seed(74121)
        self.projection = torch.linalg.qr(torch.randn((valid.shape[-1],VALUE_WIDTH),generator=rng)).Q
        projected = ((valid-self.mean)/self.std)@self.projection
        self.center = projected.mean(0)
        self.scale = projected.std(0,unbiased=False).clamp_min(SCALE_FLOOR)
        self.slots = torch.arange(VALUE_WIDTH)<AMPLITUDE_WIDTH

    def __call__(self, hidden):
        values = ((hidden.float()-self.mean)/self.std)@self.projection
        valid = values.isfinite() & self.slots
        values = ((values-self.center)/self.scale).masked_fill(~valid,0)
        return torch.cat((values,valid.to(values.dtype)),-1)


def mathematical_audit(models, inputs, logits, labels, people, optimizer):
    z = torch.tensor([1.,0.,-1.,0.,.2,1.], dtype=torch.float64)
    y = torch.tensor([1,4,2,4,3,0])
    subjects = torch.tensor([0,0,1,1,1,2])
    for local in (False,True):
        value,count,participants = rank_loss(z,y,subjects,local)
        same,_,_ = rank_loss(z+2,y,subjects,local)
        improved,_,_ = rank_loss(z+(y<4)*.2-(y==4)*.2,y,subjects,local)
        torch.testing.assert_close(value,same,rtol=0,atol=1e-15)
        assert improved<value and (count,participants)==((3,2) if local else (6,2))
    expected = (F.softplus(z[1]-z[0])+F.softplus(z[3]-z[[2,4]]).mean())/2
    torch.testing.assert_close(rank_loss(z,y,subjects,True)[0],expected,rtol=0,atol=0)
    for yl,ids in ((y.clamp_max(3),subjects),(y,torch.arange(len(y)))):
        missing,count,participants = rank_loss(z,yl,ids,True)
        assert missing.item()==count==participants==0
    with torch.no_grad():
        affine = Calibration(True)
        affine.theta.fill_(-.7)
        affine.bias.fill_(.3)
        assert torch.equal(torch.argsort(z,stable=True),torch.argsort(affine(None,z),stable=True))
    reference = track.five_class_log_probs(*logits).detach()
    initial = models['R03'].state_dict()
    for arm,model in models.items():
        if arm in ('R03','R04','R05'):
            assert sum(v.numel() for v in model.parameters())==24*inputs.shape[-1]+49
            for name,value in model.state_dict().items():
                torch.testing.assert_close(value,initial[name],rtol=0,atol=0)
        got = track.five_class_log_probs(logits[0],model(inputs,logits[1]),logits[2])
        torch.testing.assert_close(got,reference,rtol=0,atol=0)
        if arm in ('R04','R05'):
            state = {k:v.clone() for k,v in model.state_dict().items()}
            opt = torch.optim.AdamW(model.parameters(),**optimizer)
            use = slice(0,track.BATCH)
            loss,_ = train_loss(model,arm,inputs[use],tuple(v[use] for v in logits),labels[use],people[use])
            loss.backward()
            assert model.layers[-1].weight.grad.abs().max()>0
            opt.step()
            opt.zero_grad(set_to_none=True)
            loss,_ = train_loss(model,arm,inputs[use],tuple(v[use] for v in logits),labels[use],people[use])
            loss.backward()
            assert model.layers[0].weight.grad.abs().max()>0
            assert all(v.grad is None and not v.requires_grad for v in (*logits,inputs))
            model.load_state_dict(state)
            model.zero_grad(set_to_none=True)
    return ['zero_delta_and_shared_initialization','positive_slope_and_stable_order',
            'rank_shift_invariance_and_direction', 'no_pair_and_subject_equal_pairing',
            'nonzero_residual_gradient_and_frozen_inputs']


def read_record(db,key):
    row = db.execute('SELECT value FROM records WHERE key=?',(key,)).fetchone()
    return row[0] if row else None


def write_record(db,key,value):
    with db:
        db.execute('INSERT OR REPLACE INTO records VALUES (?,?)',(key,value))


@torch.no_grad()
def native_audit(ref, schedule, training):
    ref.model.requires_grad_(False).eval()
    for p in ref.params.values():
        p.requires_grad_(True)
    before = ref.weights()
    x,y = ref.batch('fit',schedule[:32])
    ref.model.eval()
    def read(wave):
        captured = []
        handle = ref.model.base.decoder.register_forward_pre_hook(lambda _,args: captured.append(torch.cat(args,-1)))
        try:
            out = ref.model(wave,ref.model.base.fold_local_features(wave))
        finally:
            handle.remove()
        assert len(captured)==1
        return (captured[0],*out.conditional_logits)
    same = read(x)
    for expected,actual in zip(training,same,strict=True):
        torch.testing.assert_close(expected[:32],actual.cpu(),atol=audit.ATOL,rtol=0)
    rng = torch.Generator(device=ref.device).manual_seed(74121)
    causality = []
    for i,stop in enumerate(repro.POSITIONS):
        changed = x.clone()
        changed[...,stop:] = 2*torch.randn(changed[...,stop:].shape,device=ref.device,generator=rng)
        check = read(changed)
        gap = 0.
        for base,out in zip(same,check,strict=True):
            torch.testing.assert_close(base[:,:i+1],out[:,:i+1],atol=audit.ATOL,rtol=0)
            gap = max(gap,float((base[:,:i+1]-out[:,:i+1]).abs().max()))
        causality.append(gap)
    for name,p in ref.weights().items():
        torch.testing.assert_close(p,before[name],atol=0,rtol=0)
    ref.model.requires_grad_(False)
    assert all(p.grad is None and not p.requires_grad for p in ref.model.parameters())
    return y.cpu(),causality


def fit_groups(models, db, key, optimizer, h, logits, labels, people, limit):
    fits = {}
    for arm,model in models.items():
        started = monotonic()
        opt = torch.optim.AdamW(model.parameters(),**optimizer)
        path = key+'/'+arm+'/fit'
        old = read_record(db,path)
        log = []
        if old is not None:
            saved = torch.load(io.BytesIO(old),weights_only=True)
            model.load_state_dict(saved['model'])
            opt.load_state_dict(saved['optimizer'])
            log = saved['log']
        for step in range(len(log),STEPS):
            limit()
            index = slice(step*track.BATCH,(step+1)*track.BATCH)
            logit = tuple(t[index] for t in logits)
            loss,row = train_loss(model,arm,h[index],logit,labels[index],people[index])
            assert bool(torch.isfinite(loss))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            log.append(dict(step=step+1,**row))
            if (step+1)%32==0:
                save = io.BytesIO()
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),log=log),save)
                write_record(db,path,save.getvalue())
        fits[arm] = dict(steps=log,effective_steps=sum(row['pairs']>0 for row in log),
                         total_pairs=sum(row['pairs'] for row in log),
                         parameters=sum(v.numel() for v in model.parameters()),seconds=monotonic()-started)
        if arm=='R02':
            fits[arm].update(slope=float(model.theta.detach().double().exp()),bias=float(model.bias.detach()))
        if arm=='R01':
            fits[arm]['bias'] = float(model.bias.detach())
        model.eval()
        print(f'{arm} 完成 128 步，耗时 {fits[arm]["seconds"]:.2f}s',flush=True)
    return fits


def event_values(p, positive, runs, native=False):
    weights = p if native else p.astype(np.float64)
    cumulative = np.vstack((np.zeros(4),np.cumsum(weights[:,1:]*positive[:,None],axis=0)))
    return cumulative[runs[:,1]]-cumulative[runs[:,0]]


def event_details(y,p,positive,matches,counts):
    alarms = b07.runs(positive.astype(int),gap=5,min_positive=5)
    posterior = event_values(p,positive,alarms,native=True)
    matched = np.isfinite(matches[:,3])
    ids = np.searchsorted(alarms[:,0],(matches[matched,0]+matches[matched,4]).astype(int))
    true = matches[matched,2].astype(int)
    selected = posterior[ids]
    subset = np.isin(true,[1,2,3])
    best_q = selected[:,:3].argmax(1)+1
    wrong_b = selected[np.arange(len(true)),(true-1)]<=selected[:,3]
    suppressed = int((subset & (best_q==true) & wrong_b).sum())
    score = event_values(p,positive,alarms)[ids]
    mass = score.sum(1)
    b = np.divide(score[:,:3].sum(1),mass,out=np.full(len(score),np.nan),where=mass>0)
    valid = np.isfinite(b)
    pred = selected.argmax(1)+1
    np.testing.assert_array_equal(np.bincount(pred[subset],minlength=5),np.asarray(counts['event_cm'])[:3,:].sum(0)-[counts['truth']-counts['detected']-counts['event_cm'][3][0],0,0,0,0])
    return (true[valid]<4,b[valid]),suppressed


def summarize_counts(counts):
    ordinary = np.asarray(counts['event_cm'])
    counts['event_metrics'] = {}
    for j,name in enumerate(b07.CLASS_NAMES[1:]):
        tp = int(ordinary[j,j+1])
        pred = counts['alarms_by_cls'][j]
        truth = int(ordinary[j].sum())
        counts['event_metrics'][name] = dict(tp=tp,pred=pred,truth=truth,precision=tp/pred if pred else 0.,
                                            recall=tp/truth if truth else 0.,f1=2*tp/(pred+truth) if pred+truth else 0.)
    return counts


def binary_measures(y,b,z=None):
    if len(y)==0:
        return dict(auroc=None,auprc=None,brier=None,nll=None,ece=None,n=0)
    estimate = b if z is None else z
    if z is None:
        safe = np.clip(b,np.finfo(np.float64).eps,1-np.finfo(np.float64).eps)
        z = np.log(safe)-np.log1p(-safe)
    _, predicted, observed = r0.probability_bins(y,b)
    ece = np.abs(predicted-observed).sum()/len(y)
    return dict(auroc=float(roc_auc_score(y,estimate)) if np.unique(y).size==2 else None,
                auprc=float(average_precision_score(y,estimate)) if y.any() else None,
                brier=float(np.square(b-y).mean()), nll=float(np.logaddexp(0,np.where(y,-z,z)).mean()),
                ece=float(ece),n=len(y))


def one_night(y,p,reference):
    counts,positive,matches = b07.event_score(y,p)
    prior,prior_pos,prior_match = reference
    np.testing.assert_array_equal(positive,prior_pos)
    np.testing.assert_array_equal(matches[:,[0,1,2,4,5]],prior_match[:,[0,1,2,4,5]])
    assert all(counts[k]==prior[k] for k in ('seconds','predicted','truth','detected','iou_sum'))
    cm = track.confusion(y,p.argmax(1))
    ab = p[:,1:].astype(np.float64).sum(1)
    ok = (y>0)&(ab>0)
    apnea = y[ok]<4
    conditional = p[ok,1:4].astype(np.float64).sum(1)/ab[ok]
    event,suppressed = event_details(y,p,positive,matches,counts)
    return dict(counts=summarize_counts(counts),cm=cm.tolist(),q_correct_b_blocked=suppressed,
                undefined_b=int(((y>0)&(ab==0)).sum()),binary=(apnea,conditional),event=event)


def aggregate(rows):
    totals = {k:sum(row['counts'][k] for row in rows) for k in ('seconds','predicted','truth','detected','typed')}
    for k in ('event_cm','alarms_by_cls'):
        totals[k] = np.sum([row['counts'][k] for row in rows],axis=0).tolist()
    y,b = (np.concatenate(parts) for parts in zip(*(r['binary'] for r in rows)))
    logits = np.concatenate([r['logit'] for r in rows])
    binary = binary_measures(y,b,logits)
    event = binary_measures(*(np.concatenate(parts) for parts in zip(*(r['event'] for r in rows))))
    auc = [row['measures']['auroc'] for row in rows if row['measures']['auroc'] is not None]
    totals = summarize_counts(totals)
    return dict(**totals, subject_equal_auroc=float(np.mean(auc)) if auc else None, valid_auroc_subjects=len(auc),
                pooled=binary,fixed_alarm_auroc=event['auroc'],fixed_alarm_subject_events=event['n'],
                macro_f1=track.macro_f1(np.sum([row['cm'] for row in rows],axis=0)),
                detect_f1=2*totals['detected']/(totals['predicted']+totals['truth']),
                fa_h=3600*(totals['predicted']-totals['detected'])/totals['seconds'],
                fixed_f1=2*totals['typed']/(totals['predicted']+totals['truth']),
                apnea_to_hyp=sum(totals['event_cm'][j][4] for j in range(3)),
                hyp_to_apnea=sum(totals['event_cm'][3][1:4]),
                q_correct_b_blocked=sum(r['q_correct_b_blocked'] for r in rows))


def reconstruct(parent, new_logit, cached_q):
    p = parent.astype(np.float64,copy=True)
    mass = p[:,1:].sum(1)
    pause = p[:,1:4].sum(1)
    valid = (mass>0)&(pause>0)
    np.testing.assert_allclose(softmax(cached_q[valid].astype(np.float64),axis=1),
                               p[valid,1:4]/pause[valid,None],rtol=0,atol=3e-6)
    q = p[valid,1:4]/pause[valid,None]
    b = expit(new_logit[valid])
    p[valid,1:4] = (mass[valid]*b)[:,None]*q
    p[valid,4] = mass[valid]-p[valid,1:4].sum(1)
    np.testing.assert_array_equal(p[:,0],parent[:,0])
    np.testing.assert_allclose(p[:,1:].sum(1),mass,rtol=0,atol=5e-16)
    np.testing.assert_allclose(p[valid,1:4]/p[valid,1:4].sum(1)[:,None],q,rtol=0,atol=5e-15)
    return p,dict(undefined_q=int((pause==0).sum()),undefined_b=int((mass==0).sum()))


@torch.no_grad()
def score_groups(saved, roles, labels, norm, models, ref, provenance, limit):
    rows = {arm:{} for arm in ARMS}
    checks = {}
    for person in roles['select']:
        limit()
        target,clock = labels[person]
        y,parent = target[:,-1],saved[person]
        audit.check_predictions(parent,saved[person+'_second'],clock)
        order = np.column_stack((np.full(len(y),roles['select'].index(person)),np.arange(len(y))))
        native,cache_hash = repro.encoding_cache(ref,provenance,'select',order,1,person,limit)
        hidden,a,b,q = (v[:,0] for v in native)
        replay = track.five_class_log_probs(a,b,q).exp().numpy()
        np.testing.assert_allclose(replay,parent,atol=audit.ATOL,rtol=0)
        model_input = norm(hidden)
        reference = b07.event_score(y,parent)
        info = dict(cache_sha256=cache_hash,probability_replay=float(np.max(np.abs(replay-parent))),seconds=len(y))
        for arm in ARMS:
            predicted = parent
            z_out = b.double().numpy()
            if arm!='R00':
                if arm=='R02':
                    z_out = models[arm].theta.double().exp().item()*z_out+models[arm].bias.item()
                    np.testing.assert_array_equal(np.argsort(b.numpy(),kind='stable'),np.argsort(z_out,kind='stable'))
                else:
                    z_out = models[arm](model_input,b).double().numpy()
                predicted,undefined = reconstruct(parent,z_out,q.numpy())
                info[arm] = undefined
            audit.check_predictions(predicted,clock,clock)
            row = one_night(y,predicted,reference)
            valid_b = (y>0)&(predicted[:,1:].sum(1)>0)
            row['measures'] = binary_measures(*row['binary'],z=z_out[valid_b])
            row['logit'] = z_out[valid_b]
            rows[arm][person] = row
        assert rows['R02'][person]['measures']['auroc']==rows['R00'][person]['measures']['auroc']
        checks[person] = info
        print(f'{person} 完成逐夜概率与报警核验',flush=True)
    return rows,checks


def paired_bootstrap(rows, groups):
    names = tuple(rows['R00'])
    draws = np.random.default_rng(74121).integers(len(names),size=(BOOT,len(names)))
    out = {}
    for arm,base in (('R02','R00'),('R03','R00'),('R04','R03'),('R05','R03'),('R05','R04')):
        high,low = ([rows[key][p] for p in names] for key in (arm,base))
        stats = {}
        auc_diff = np.array([(h['measures']['auroc']-l['measures']['auroc'])
                             if h['measures']['auroc'] is not None and l['measures']['auroc'] is not None else np.nan
                             for h,l in zip(high,low,strict=True)])
        sample = auc_diff[draws]
        count = np.isfinite(sample).sum(1)
        stats['subject_equal_auroc'] = np.divide(np.nansum(sample,axis=1),count,
                                                   out=np.full(BOOT,np.nan),where=count>0)
        def select(group, func):
            return np.asarray([func(r) for r in group])[draws]
        def boost(group):
            typed = select(group,lambda r:r['counts']['typed']).sum(1)
            div = select(group,lambda r:r['counts']['predicted']+r['counts']['truth']).sum(1)
            brier = select(group,lambda r:(r['measures']['brier'] or 0)*r['measures']['n']).sum(1)
            n = select(group,lambda r:r['measures']['n']).sum(1)
            matrix = select(group,lambda r:r['cm']).sum(1)
            denom = matrix.sum(1)+matrix.sum(2)
            return dict(fixed_f1=2*typed/div,brier=brier/n,
                        macro_f1=np.divide(2*matrix.diagonal(axis1=1,axis2=2),denom,
                                           out=np.zeros_like(denom,dtype=float),where=denom>0).mean(1))
        h,l = boost(high),boost(low)
        stats.update({k:h[k]-l[k] for k in h})
        contrasts = {}
        for name,vector in stats.items():
            actual = vector[np.isfinite(vector)]
            result_h,result_l = (groups[key]['pooled']['brier'] if name=='brier' else groups[key][name] for key in (arm,base))
            contrasts[name] = dict(delta=result_h-result_l if result_h is not None and result_l is not None else None,
                                   ci95=np.quantile(actual,[.025,.975]).tolist() if len(actual) else None,
                                   valid_fraction=len(actual)/BOOT)
        net = {p:h['counts']['typed']-l['counts']['typed'] for p,h,l in zip(names,high,low,strict=True)}
        positives = sum(max(n,0) for n in net.values())
        out[arm+'-'+base] = dict(metrics=contrasts,correct_delta_by_subject=net,
                                positive_delta=int(positives),net_delta=int(sum(net.values())),
                                top_positive_share=max(net.values())/positives if positives else None,
                                subjects_auc_improved=int((auc_diff>0).sum()),subjects_auc_declined=int((auc_diff<0).sum()),
                                auc_delta_by_subject=dict(zip(names,[float(x) if np.isfinite(x) else None for x in auc_diff])))
    return out


def r06_diagnostic(roster, labels, prior, limit):
    result, sources = {}, {}
    for seed in track.SEEDS:
        result[str(seed)] = {}
        for fold,roles in roster.items():
            limit()
            _,_,arrays,_ = repro.t03_run(fold,seed)
            checksum = audit.digest(arrays)
            assert checksum==prior['archive']['archives'][f'{fold}/{seed}']['arrays_sha256']
            sources[f'{fold}/{seed}'] = checksum
            with np.load(io.BytesIO(arrays)) as saved:
                for person in roles['select']:
                    target,clock = labels[person]
                    y,p = target[:,-1],saved[person]
                    audit.check_predictions(p,saved[person+'_second'],clock)
                    counts,positive,matches = b07.event_score(y,p)
                    alarms = b07.runs(positive.astype(int),gap=5,min_positive=5)
                    scores = event_values(p,positive,alarms,native=True)
                    choose = np.where(scores[:,:3].sum(1)>scores[:,3],scores[:,:3].argmax(1)+1,4)
                    matched = np.isfinite(matches[:,3])
                    ids = np.searchsorted(alarms[:,0],(matches[matched,0]+matches[matched,4]).astype(int))
                    original = scores.argmax(1)+1
                    np.testing.assert_array_equal(original[ids],matches[matched,3])
                    true = matches[:,2].astype(int)
                    choices = np.zeros(len(matches),dtype=int)
                    choices[matched] = choose[ids]
                    cm = np.zeros((4,5),dtype=int)
                    np.add.at(cm,(true-1,choices),1)
                    counts_new = dict(counts,typed=int((true==choices).sum()),event_cm=cm.tolist(),
                                      alarms_by_cls=np.bincount(choose,minlength=5)[1:].tolist(),
                                      apnea_to_hyp=int(cm[:3,4].sum()))
                    result[str(seed)][person] = dict(original=summarize_counts(counts),hierarchical=summarize_counts(counts_new))
    def roll(people):
        out = {}
        for k in ('original','hierarchical'):
            ts = {field:sum(l[k][field] for l in people) for field in ('typed','truth','detected','predicted','seconds')}
            ts.update(event_cm=np.sum([l[k]['event_cm'] for l in people],axis=0).tolist(),
                      alarms_by_cls=np.sum([l[k]['alarms_by_cls'] for l in people],axis=0).tolist())
            ts['fixed_f1'] = 2*ts['typed']/(ts['truth']+ts['predicted'])
            out[k] = summarize_counts(ts)
        return out
    assert all(len(people)==20 for people in result.values())
    by_seed = {seed:roll(list(people.values())) for seed,people in result.items()}
    return dict(by_seed=by_seed, aggregate=roll([r for people in result.values() for r in people.values()]),
                subjects=result,archives=sources)


def engineering(ref, prior, limit):
    historical = prior['batch']['methods']['fp32_vs_native_same_batch'] if prior['batch'] is not None else None
    start = monotonic()
    backbone = ref.model.base.backbone
    backbone.llm.float()
    backbone.fixed_prefix = backbone.fixed_prefix.float()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    # Preserve the original eval dispatch flags as in R0. All forwards use inference mode.
    for parameter in ref.params.values():
        parameter.requires_grad_(True)
    ref.model.eval()
    subject,start_row = ref.train[0]
    person = ref.data.roles['fit'][subject]
    clock = ref.data.ends[person]-1
    start_row = min(start_row,len(clock)-256)
    selected = np.arange(start_row,start_row+256)
    assert np.all(np.diff(clock[selected])==1)
    schedule = np.column_stack((np.full(256,subject),selected))
    x,_ = ref.batch('fit',schedule)
    result = dict(historical_fp32_vs_native=historical,windows=256,positions=list(repro.POSITIONS),warmups=10,
                  schedule_sha256=audit.digest(schedule.tobytes()),person=person,clock=clock[selected].tolist(),backend='math SDPA',
                  matmul=torch.get_float32_matmul_precision(),cuda=torch.version.cuda,
                  device=torch.cuda.get_device_name(ref.device) if ref.device.type=='cuda' else str(ref.device),cudnn_tf32=torch.backends.cudnn.allow_tf32,
                  tf32=torch.backends.cuda.matmul.allow_tf32,mha_fastpath=torch.backends.mha.get_fastpath_enabled())
    on_cuda = ref.device.type=='cuda'
    def sync():
        if on_cuda:
            torch.cuda.synchronize(ref.device)
    with torch.inference_mode(),torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        output = {}
        for size in (1,32):
            for _ in range(10):
                limit()
                ref.model(x[:size])
            sync()
            if on_cuda:
                torch.cuda.reset_peak_memory_stats(ref.device)
            elapsed, blocks = [], []
            for section in x.split(size):
                limit()
                sync()
                t = perf_counter()
                logp = ref.model(section).log_prob
                sync()
                elapsed.append((perf_counter()-t)*1000)
                blocks.append(logp.float().cpu().numpy())
            output[size] = np.concatenate(blocks)
            result[f'batch_{size}'] = dict(median_ms=float(np.median(elapsed)),p95_ms=float(np.quantile(elapsed,.95)),
                                          max_memory_allocated_bytes=torch.cuda.max_memory_allocated(ref.device) if on_cuda else None,
                                          max_memory_reserved_bytes=torch.cuda.max_memory_reserved(ref.device) if on_cuda else None)
    compare = lambda l,r:r0.compare_layers({'probability':np.exp(l),'log probability':l},
                                           {'probability':np.exp(r),'log probability':r})
    result['all_positions'] = compare(output[32],output[1])
    result['latest'] = compare(output[32][:,-1:],output[1][:,-1:])
    stats = result['all_positions']
    result['pass'] = (stats['layers']['probability']['max_abs']<=1e-4 and
                      stats['layers']['log probability']['max_abs']<=1e-3 and
                      stats['decision_flips']==stats['alarm_flips']==0)
    result['seconds'] = monotonic()-start
    return result


def acceptance(groups, contrasts, fits):
    checks = {}
    for arm in ('R04','R05'):
        compared = contrasts[arm+'-R03']
        delta = {k:v['delta'] for k,v in compared['metrics'].items()}
        checks[arm] = dict(subject_auc=delta['subject_equal_auroc']>=.01,fixed_f1=delta['fixed_f1']>=.01,
                           brier=delta['brier']<=.005,macro_f1=delta['macro_f1']>=-.005,
                           detection_and_fa=all(groups[arm][k]==groups['R03'][k] for k in ('detect_f1','fa_h')),
                           coverage=arm!='R05' or fits[arm]['effective_steps']>=32,
                           top_positive_share=compared['top_positive_share'])
        row = checks[arm]
        row['passes_numeric_conditions'] = all(row[k] for k in ('subject_auc','fixed_f1','brier','macro_f1','detection_and_fa','coverage'))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--started-at',type=float,default=monotonic())
    args = parser.parse_args()
    def limit():
        if monotonic()-args.started_at>=600:
            raise TimeoutError('整批已到 600 秒软限')
    limit()
    signal.alarm(max(1,int(900-(monotonic()-args.started_at))))
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    phases = {}
    start = monotonic()
    r0path = r0.OUT
    prior = json.loads(r0path.read_text())
    r0_hash = track.hash_file(r0path)
    assert prior['T03']==audit.VERSION and prior['status']=='completed'
    roster = repro.roster()
    roles = roster[1]
    labels = audit.reference_labels(roster)
    r06 = r06_diagnostic(roster,labels,prior,limit)
    phases['archive_diagnostic'] = monotonic()-start
    start = monotonic()
    ref,provenance,replay = repro.reference()
    for parameter in ref.params.values():
        parameter.requires_grad_(True)
    schedule = ref.train[:STEPS*track.BATCH]
    expected = audit.digest(schedule.tobytes())
    training,fit_cache_hash = repro.encoding_cache(ref,provenance,'fit',schedule,5,'fit',limit)
    y = torch.from_numpy(np.stack([labels[roles['fit'][i]][0][j] for i,j in schedule])).long()
    observed,causal = native_audit(ref,schedule,training)
    torch.testing.assert_close(y[:32],observed,atol=0,rtol=0)
    subject = torch.as_tensor(schedule[:,0,None]).expand_as(y)
    hidden,*logits = training
    norm = HiddenInput(hidden,y)
    inputs = norm(hidden)
    models = dict(R01=Calibration(False),R02=Calibration(True),R03=Residual(inputs.shape[-1]),
                  R04=Residual(inputs.shape[-1]),R05=Residual(inputs.shape[-1]))
    optimizer = repro.OPT
    tests = mathematical_audit(models,inputs,logits,y,subject,optimizer)
    phases['cache_loading_and_audit'] = monotonic()-start
    config = dict(fold=1,seed=SEED,steps=STEPS,batch=track.BATCH,windows=len(schedule),
                  inputs={'raw_hidden':hidden.shape[-1],'residual':inputs.shape[-1],'active_slots':int(norm.slots.sum())*2},
                  rank_weight=RANK_WEIGHT,scale_floor=SCALE_FLOOR,optimizer=optimizer,clip_grad_norm=1.,
                  train_index=expected,fit_cache_sha256=fit_cache_hash,
                  parameter_names=list(models['R03'].state_dict()),source=track.hash_file(Path(__file__)),
                  early_stop=False,bootstrap=dict(samples=BOOT,seed=74121,unit='subject'))
    identity = dict(T03=provenance,config=config,r0=r0_hash)
    fit_identity = dict(config={k:v for k,v in config.items() if k!='source'}, provenance=provenance,
                        fit_implementation=fit_fingerprint())
    key = 'BreathLLM-R1/1/4121/'+audit.digest(json.dumps(fit_identity,sort_keys=True).encode())[:24]
    print('R06 已完成，缓存身份、数学和五个因果位置核验通过',flush=True)
    start = monotonic()
    with closing(sqlite3.connect(RECORDS,timeout=15)) as db:
        db.execute('CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, value BLOB)')
        stats = io.BytesIO()
        torch.save(dict(mean=norm.mean,std=norm.std,projection=norm.projection,center=norm.center,
                        scale=norm.scale,slots=norm.slots,train_index=torch.from_numpy(schedule)),stats)
        write_record(db,key+'/normalization',stats.getvalue())
        fits = fit_groups(models,db,key,optimizer,inputs,logits,y,subject,limit)
        phases['fitting'] = monotonic()-start
        start = monotonic()
        with np.load(io.BytesIO(ref.arrays)) as saved:
            rows,night_checks = score_groups(saved,roles,labels,norm,models,ref,provenance,limit)
        groups = {k:aggregate(list(v.values())) for k,v in rows.items()}
        contrasts = paired_bootstrap(rows,groups)
        phases['scoring_and_bootstrap'] = monotonic()-start
        assert groups['R02']['pooled']['auroc']==groups['R00']['pooled']['auroc']
        for members in rows.values():
            for row in members.values():
                for name in ('binary','event','logit'):
                    row.pop(name)
        result = dict(status='probes_completed',archive_key=key,identity=identity,tests=tests,causal_max_abs=causal,
                      replay=replay,fit=fits,groups=groups,nights=night_checks,subjects=rows,
                      comparisons=contrasts,acceptance=acceptance(groups,contrasts,fits),R06=r06,phases=phases)
        seal(result)
        print('整夜评分、逐人结果和 Bootstrap 已保存',flush=True)
        try:
            result['fp32'] = engineering(ref,prior,limit)
            result['status'] = 'completed'
        except TimeoutError as error:
            result['fp32'] = dict(status='partial',reason=str(error))
        result['seconds'] = monotonic()-args.started_at
        payload = seal(result)
        write_record(db,key+'/report',payload)
        print(f'R1 状态 {result["status"]}，耗时 {result["seconds"]:.2f}s',flush=True)
    signal.alarm(0)


def fit_fingerprint():
    import inspect
    return audit.digest(*(inspect.getsource(obj).encode() for obj in
                          (Residual,Calibration,HiddenInput,rank_loss,train_loss,fit_groups)))


def seal(result):
    body = json.dumps(result,ensure_ascii=False,allow_nan=False).encode()
    repro.atomic(OUT,lambda stream:stream.write(body))
    return body


if __name__=='__main__':
    main()
