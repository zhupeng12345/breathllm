"""Verify the sealed Track A runs and compute subject-level paired effects."""

from collections import defaultdict

import numpy as np

import B07基座改造实验 as experiment


track = experiment.track
COUNTS = ("seconds","predicted","truth","detected","typed","iou_sum","q_blocked","matched_apnea","apnea_to_hyp")
NAMES = ("Macro-F1","Event F1","Typed F1","FA/h","检出率","Boundary IoU","Accuracy","Balanced Acc",
         "OSA事件召回","CSA事件召回","MSA事件召回","低通气事件召回","q等权召回","暂停判低通气","q受限")


def vector(row, diag):
    return np.array([*np.ravel(diag["cm"]),*(row[k] for k in COUNTS),
                     *np.ravel(row["event_cm"]),*row["alarms_by_cls"],*np.ravel(diag["q_cm"])])


def rate(num,den):
    return num/np.maximum(den,1e-10)


def metrics(x):
    cm=x[..., :25].reshape(*x.shape[:-1],5,5)
    tp=np.diagonal(cm,axis1=-2,axis2=-1)
    s,p,r,d,t,iou,q_block,q_support,as_hyp=np.moveaxis(x[...,25:34],-1,0)
    ec=x[...,34:54].reshape(*x.shape[:-1],4,5)
    qc=x[...,58:67].reshape(*x.shape[:-1],3,3)
    correct=np.diagonal(ec[...,1:],axis1=-2,axis2=-1)
    out=[rate(2*tp,cm.sum(-2)+cm.sum(-1)).mean(-1),rate(2*d,p+r),rate(2*t,p+r),
         rate(3600*(p-d),s),rate(d,r),rate(iou,d),rate(tp.sum(-1),s),rate(tp,cm.sum(-1)).mean(-1),
         *np.moveaxis(rate(correct,ec.sum(-1)),-1,0),
         rate(np.diagonal(qc,axis1=-2,axis2=-1),qc.sum(-1)).mean(-1),rate(as_hyp,q_support),rate(q_block,q_support)]
    return np.stack(out,axis=-1)


def class_metrics(counts):
    cm = counts[:25].reshape(5,5)
    events = counts[34:54].reshape(4,5)
    q = counts[58:67].reshape(3,3)
    seconds_correct = cm.diagonal()
    event_correct = events[:,1:].diagonal()
    alarm_count = counts[54:58]
    return dict(
        second=dict(confusion=cm.tolist(), precision=rate(seconds_correct,cm.sum(0)).tolist(),
                    recall=rate(seconds_correct,cm.sum(1)).tolist(),
                    f1=rate(2*seconds_correct,cm.sum(0)+cm.sum(1)).tolist(),
                    distribution=rate(cm.sum(0),cm.sum()).tolist()),
        event=dict(confusion=events.tolist(), support=events.sum(1).tolist(),
                   detected=events[:,1:].sum(1).tolist(), missed=events[:,0].tolist(),
                   correct=event_correct.tolist(), alarm_count=alarm_count.tolist(),
                   precision=rate(event_correct,alarm_count).tolist(),
                   recall=rate(event_correct,events.sum(1)).tolist(),
                   f1=rate(2*event_correct,alarm_count+events.sum(1)).tolist(),
                   matched_accuracy=rate(event_correct.sum(),events[:,1:].sum()).item(),
                   confirmation_delay=None),
        q=dict(confusion=q.tolist(), recall=rate(q.diagonal(),q.sum(1)).tolist(),
               macro_f1=rate(2*q.diagonal(),q.sum(0)+q.sum(1)).mean().item()))


def index_subjects(rows,roster):
    grouped=defaultdict(list)
    for row in rows:
        grouped[row[0],row[1]].append(row[2])
    return np.stack([np.mean(grouped[f,p],axis=0) for f,roles in roster.items() for p in roles["select"]])


def comparisons(values,roster,reference="R00"):
    totals={g:metrics(x.sum(0)) for g,x in values.items()}
    rng=np.random.default_rng(74121)
    end=np.cumsum([0,*[len(r["select"]) for r in roster.values()]])
    draws=np.concatenate([rng.integers(a,b,size=(10000,b-a)) for a,b in zip(end[:-1],end[1:])],axis=1)
    bootstrap={g:metrics(x[draws].sum(1)) for g,x in values.items()}
    leaveout={g:metrics(x.sum(0)-x) for g,x in values.items()}
    folds={g:np.array([metrics(x[a:b].sum(0)) for a,b in zip(end[:-1],end[1:])]) for g,x in values.items()}

    def difference(weights):
        def combine(items):
            return sum(w*items[g] for g,w in weights.items())
        point=combine(totals)
        ci=np.quantile(combine(bootstrap),[.025,.975],axis=0)
        loo=combine(leaveout)
        return dict(delta=point.tolist(),lo=ci[0].tolist(),hi=ci[1].tolist(),
                    fold=combine(folds).tolist(),loo_min=loo.min(0).tolist(),loo_max=loo.max(0).tolist())

    effects={}
    if reference in values:
        effects={f"{g}-{reference}":difference({g:1,reference:-1}) for g in values if g!=reference}
    if {"H02","H03"} <= values.keys():
        effects["H03-H02"]=difference({"H03":1,"H02":-1})
    if {"T00","T01","T02","T03"} <= values.keys():
        effects["T03-T01"]=difference({"T03":1,"T01":-1})
        effects["T03-T02"]=difference({"T03":1,"T02":-1})
        effects["prompt×initialization"]=difference({"T03":1,"T01":-1,"T02":-1,"T00":1})
    factors={tuple(f):f"R{i:02d}" for i,f in enumerate(experiment.FACTORS)}
    if not set(factors.values()) <= set(values):
        return totals,effects
    for axis,name in enumerate(("b","KD","hier")):
        weights={}
        for f,g in factors.items():
            if f[axis]:
                continue
            on=factors[f[:axis]+(1,)+f[axis+1:]]
            effects[f"{name} {g}-{on}"]=difference({g:1,on:-1})
            weights.update({g:.25,on:-.25})
        effects[name]=difference(weights)
    for i,j,title in ((0,1,"b×KD"),(0,2,"b×hier"),(1,2,"KD×hier")):
        k=3-i-j
        for condition in (0,1):
            weights={g:(1 if f[i]==f[j] else -1) for f,g in factors.items() if f[k]==condition}
            effects[f"{title} other={condition}"]=difference(weights)
    return totals,effects
