"""B07 / T03 分层读出、训练与轨道 A 评分。"""

import io
import json
import sqlite3
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F

import 轨道A实验 as track
from src.labels.ucddb_mapping import CLASS_NAMES
from src.experiment_data import TARGETS
from src.score_runs import runs
from breathllm_paths import WEIGHTS

DESIGN = track.PROJECT / "docs/T03模型设计.md"


DATABASE = track.OUTPUT / "B07基座改造结果.sqlite"


FACTORS = ((1,1,1), (0,1,1), (1,0,1), (1,1,0), (0,0,1), (0,1,0), (1,0,0), (0,0,0))


FUSION = ((1,1,1),(0,1,1),(1,0,1),(1,1,0),(1,0,0),(0,1,0),(0,0,1),(0,0,0))


OPTIONS = dict(details=True, detect_first=True, gap=5, min_run=5)


PROTOCOL = dict(outer=1, seeds=track.SEEDS, steps=track.STEPS, batch=track.BATCH, lr=track.LR,
                check=track.CHECK, patience=track.PATIENCE, selection_seed=74121,
                selection_size=track.SELECT_SIZE, threshold=.5, options=OPTIONS)


class Readout(nn.Module):
    def __init__(self, base, fusion, structure="original"):
        super().__init__()
        self.base, self.fusion, self.structure = base, fusion, structure
        self.flat = nn.Linear(base.compact_questions.shape[-1], 5) if structure == "flat" else None

    def forward(self, x, history=None, fusion_probe=False):
        local = []
        handle = None
        if self.structure == "separate":
            def capture(_layer, _inputs, output):
                local.append(output.last_hidden_state[:, -15::3].float())

            gemma = self.base.backbone.llm.get_base_model().model
            handle = gemma.register_forward_hook(capture)
        try:
            event, subtype, phys = self.base.encode_states(x, history)
        finally:
            if handle is not None:
                handle.remove()
        if self.flat is not None:
            z = self.flat(subtype).float()
            logits = condition_logits(z)
            return track.ForwardResult(z.log_softmax(-1), logits, None)
        state = subtype if self.structure == "shared" else event
        main = self.base.decoder(state, subtype)
        if local:
            main = main[0], self.base.decoder.apnea(local[0]).squeeze(-1), main[2]
        expert = self.base.physiology(phys, phys)
        if fusion_probe:
            return [track.five_class_log_probs(*(m if not use else (m+e)*.5
                                                for m,e,use in zip(main,expert,mask)))
                    for mask in FUSION]
        logits = tuple((1-w)*m + w*e if w else m for m,e,w in zip(main,expert,self.fusion))
        return track.ForwardResult(track.five_class_log_probs(*logits), logits, None)


def condition_logits(z):
    pause = z[..., 1:4]
    return z[..., 1:].logsumexp(-1)-z[..., 0], pause.logsumexp(-1)-z[..., 4], pause


def spec(group, anchor):
    if group.startswith("R"):
        fuse, kd, hier = FACTORS[int(group[1:])]
        return (.5, .5*fuse, .5), .05*kd, .3*hier, "original", "old", "old"
    fusion, kd, hier, *_ = anchor
    loss = (kd, hier)
    if group.startswith("F"):
        return tuple(.5*x for x in FUSION[int(group[1:])]), *loss, "original", "old", "old"
    if group.startswith("H"):
        return (0,0,0), 0., 0., {"H01":"flat","H02":"shared","H03":"separate"}[group], "old", "uniform"
    prompt, initial = (("old","old"),("old","random"),("new","old"),("new","random"))[int(group[1:])]
    structure = anchor[3]
    if structure=="flat" and initial=="old":
        raise ValueError("Flat readout supports T01 and T03 with a matched random head")
    return fusion, *loss, structure, prompt, initial


@torch.no_grad()
def configure(base, group_spec, initial):
    for name, value in base.named_parameters():
        if name in initial:
            value.copy_(initial[name])
    fusion, _, _, structure, prompt, init = group_spec
    base.decoder.requires_grad_(structure!="flat")
    for module, weight in zip((base.physiology.event,base.physiology.apnea,base.physiology.subtype),fusion):
        module.requires_grad_(bool(weight) and structure!="flat")
    if prompt == "new":
        # Two compact question embeddings, preserving token count and positions.
        from transformers import AutoTokenizer
        weights = WEIGHTS / "gemma-3-270m"
        tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
        questions = (
            " Is breathing normal, or is there apnea or hypopnea? Consider the degree of airflow reduction. Answer:",
            " If there is apnea, is it obstructive, central, or mixed? Consider respiratory effort. Answer:",
        )
        vectors = []
        for prompt_text in questions:
            tokens = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
            vectors.append(base.backbone.llm.get_input_embeddings()(tokens.to(base.compact_questions.device)))
        base.compact_questions.copy_(torch.stack([(x.mean(1)+x[:,-1])*.5 for x in vectors], dim=1))
    model = Readout(base, group_spec[0], structure).to(base.compact_questions.device).eval()
    if init == "uniform":
        if model.flat is not None:
            model.flat.weight.zero_()
            model.flat.bias.zero_()
        else:
            for layer in (base.decoder.event, base.decoder.apnea, base.decoder.subtype):
                layer.weight.zero_()
                layer.bias.zero_()
            base.decoder.event.bias.fill_(np.log(4))
            base.decoder.apnea.bias.fill_(np.log(3))
    if init == "random":
        mixed = base.decoder.subtype.weight[2].clone()
        mixed_bias = base.decoder.subtype.bias[2].clone()
        base.decoder.event.reset_parameters()
        base.decoder.subtype.reset_parameters()
        base.decoder.subtype.weight[2].copy_(mixed)
        base.decoder.subtype.bias[2].copy_(mixed_bias)
    return model


@torch.no_grad()
def predict(model, x, history=None):
    chunks = []
    for start in range(0, len(x), track.EVAL_BATCH):
        end = start + track.EVAL_BATCH
        h = None if history is None else tuple(z[start:end] for z in history)
        chunks.append(model(x[start:end], h).log_prob[:, -1].exp().cpu().numpy())
    return np.concatenate(chunks)


def probabilities(p):
    ab = p[:, 1:4].sum(1)
    a = p[:, 1:].sum(1)
    b = np.divide(ab, a, out=np.full_like(a,.5), where=a>0)
    q = np.divide(p[:,1:4], ab[:,None], out=np.full_like(p[:,1:4],1/3), where=ab[:,None]>0)
    return a,b,q


def diagnosis(y,p):
    a,b,q = probabilities(p)
    valid, abnormal, apnea = y>=0, y>0, (y>0)&(y<4)
    pred = p.argmax(1)
    cm = track.confusion(y,pred)
    masks = [(y==i) for i in range(5)]
    result = {key:[] for key in ("a","b","q")}
    for mask in masks:
        for key, value in zip(("a","b","q"), (a,b,q)):
            result[key].append(np.quantile(value[mask],[.1,.5,.9],axis=0).tolist() if mask.any() else None)
    for name, mask, value, target in (("a",valid,a,abnormal),("b",abnormal,b,apnea)):
        result[name+"_auroc"] = roc_auc_score(target[mask],value[mask]) if np.unique(target[mask]).size==2 else None
        result[name+"_auprc"] = average_precision_score(target[mask],value[mask]) if target[mask].any() else None
    result.update(
        b_lt_half=float((b[apnea]<.5).mean()) if apnea.any() else None,
        b_brier=float(np.mean((b[abnormal]-apnea[abnormal])**2)) if abnormal.any() else None,
        b_bacc=float(((b[apnea]>=.5).mean()+(b[y==4]<.5).mean())/2) if apnea.any() and (y==4).any() else None,
        a_recall=float((a[abnormal]>=.5).mean()) if abnormal.any() else None,
        a_fpr=float((a[y==0]>=.5).mean()) if (y==0).any() else None,
        q_cm=track.confusion(np.where(apnea,y,-1),q.argmax(1)+1)[1:4,1:4].tolist(),
        cm=cm.tolist(), macro=track.macro_f1(cm),
        recall=(cm.diagonal()/cm.sum(1).clip(1)).tolist(), predicted=cm.sum(0).tolist())
    return result


def event_score(y,p,positive=None,matches=None):
    if positive is None:
        positive = (y>=0)&(p[:,1:].sum(1)>=.5)
    alarms = runs(positive.astype(int), gap=5, min_positive=5)
    cumulative = np.vstack((np.zeros(4), np.cumsum(p[:,1:]*positive[:,None],axis=0)))
    scores = cumulative[alarms[:,1]]-cumulative[alarms[:,0]]
    alarms[:,2] = scores.argmax(1)+1
    if matches is None:
        counts, matches = track.totals(y,p,.5,**OPTIONS)
        assert counts["predicted"] == len(alarms)
    truth = matches[:, :3].astype(int)
    # Independent boundary count on the label timeline, including all misses.
    clean = np.maximum(y,0)
    starts = np.flatnonzero((clean>0)&(clean!=np.r_[0,clean[:-1]]))
    support = np.bincount(clean[starts],minlength=5)[1:]
    np.testing.assert_array_equal(support,np.bincount(truth[:,2],minlength=5)[1:])
    match = np.isfinite(matches[:,3])
    matched = truth[match]
    begins = (matched[:,0]+matches[match,4]).astype(int)
    ids = np.searchsorted(alarms[:,0],begins)
    np.testing.assert_array_equal(alarms[ids,:2],matched[:,:2]+matches[match,4:6])
    guess = alarms[ids,2]
    cm = np.zeros((4,5),dtype=int)
    np.add.at(cm,(truth[~match,2]-1,0),1)
    np.add.at(cm,(matched[:,2]-1,guess),1)
    inter = np.minimum(alarms[ids,1],matched[:,1])-np.maximum(begins,matched[:,0])
    union = np.maximum(alarms[ids,1],matched[:,1])-np.minimum(begins,matched[:,0])
    subtype = matched[:,2]<4
    matched_scores = scores[ids[subtype]]
    row = dict(seconds=int((y>=0).sum()),predicted=len(alarms),truth=len(truth),detected=int(match.sum()),
               typed=int((guess==matched[:,2]).sum()),event_cm=cm.tolist(),iou_sum=float((inter/union).sum()),
               alarms_by_cls=np.bincount(alarms[:,2],minlength=5)[1:].tolist(),
               q_blocked=int((matched_scores[:,:3].sum(1)<=matched_scores[:,3]).sum()),
               matched_apnea=int(subtype.sum()),apnea_to_hyp=int(((guess==4)&subtype).sum()))
    return row, positive, matches


def ideals(y,p):
    a,b,q = probabilities(p)
    is_apnea = (y>0)&(y<4)
    perfect_b = np.where(y>0,is_apnea,b)
    perfect_q = np.where(is_apnea[:,None],np.eye(3)[(y-1).clip(0,2)],q)
    for name, bb, qq in (("D01",perfect_b,q),("D02",b,perfect_q),("D03",perfect_b,perfect_q)):
        altered = np.column_stack((p[:,0],(a*bb)[:,None]*qq,a*(1-bb)))
        altered[y<=0] = p[y<=0]
        np.testing.assert_allclose(altered[:,1:].sum(1),a,atol=2e-7,rtol=0)
        yield name, altered


@torch.no_grad()
def prepare_history(base,x,size):
    chunks = [base.fold_local_features(s) for s in x.split(size)]
    return tuple(torch.cat(part) for part in zip(*chunks))


def training_stats(output,y,teacher):
    a,b,q = output.conditional_logits
    safe = y.clamp_min(0)
    is_apnea = (y>0)&(y<4)
    nll = -output.log_prob.gather(-1,safe[...,None]).squeeze(-1)
    columns = [nll,F.binary_cross_entropy_with_logits(a,(y>0).float(),reduction="none"),
               F.binary_cross_entropy_with_logits(b,is_apnea.float(),reduction="none")*(y>0),
               F.cross_entropy(q.flatten(0,1),(y-1).clamp(0,2).flatten(),reduction="none").reshape_as(y)*is_apnea]
    if teacher is not None:
        columns.append(4*F.kl_div((output.log_prob/2).log_softmax(-1),
                                   (teacher/2).softmax(-1),reduction="none").sum(-1))
    columns += [t.grad.detach().reshape(*y.shape,-1).square().sum(-1) for t in (a,b,q)]
    onehot = F.one_hot(safe,5).float()*(y>=0)[...,None]
    stats = onehot.flatten(0,1).T @ torch.stack(columns,-1).flatten(0,1)
    return torch.column_stack((onehot.sum((0,1)),onehot.any(1).any(0),stats))


def train(model, params, tx, ty, th, teacher, vx, vy, vh, kd, hier, *, resume=None, checkpoint=None):
    opt = torch.optim.AdamW(params.values(),lr=track.LR)
    clock = perf_counter()
    best, best_step, curve, elapsed = -1., 0, [], 0.
    labels = vy[:,-1].cpu().numpy()
    snapshots = {}
    first_step = 0
    if resume is not None:
        opt.load_state_dict(resume["optimizer"])
        with torch.no_grad():
            for name,value in params.items():
                value.copy_(resume["weights"][name])
        torch.random.set_rng_state(resume["rng"])
        if tx.is_cuda:
            torch.cuda.set_rng_state(resume["cuda_rng"],tx.device)
        best, best_step, elapsed = resume["best"],resume["best_step"],resume["elapsed"]
        curve,snapshots = json.loads(resume["curve"]),resume["snapshots"]
        first_step = track.STEPS+1 if resume["stopped"] else resume["step"]+track.CHECK
    for step in range(first_step,track.STEPS+1,track.CHECK):
        totals, loss_sum = None, 0.
        start = perf_counter()
        first = max(0,step-track.CHECK)
        last = step if step else 1
        for i in range(first,last):
            sl = slice(i*track.BATCH,(i+1)*track.BATCH)
            y = ty[sl]
            output = model(tx[sl],tuple(z[sl] for z in th))
            for logit in output.conditional_logits:
                logit.retain_grad()
            guidance = None if kd==0 else teacher[sl]
            loss = track.classification_loss(output,y,teacher_logits=guidance,hierarchy_weight=hier,distillation_weight=kd)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                new = training_stats(output,y,guidance)
                totals = new if totals is None else totals+new
                loss_sum = loss_sum+loss.detach()
            if step:
                nn.utils.clip_grad_norm_(params.values(),1.)
                opt.step()
        stat = totals.cpu().numpy()
        loss_value = float(loss_sum)/(last-first)
        if not np.isfinite(stat).all() or not np.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite training diagnostics at step {step}")
        elapsed += perf_counter()-start
        d = diagnosis(labels,predict(model,vx,vh))
        curve.append(dict(step=step, avg_loss=loss_value, train=stat.tolist(),
                          seconds=elapsed, validation=d))
        snapshots[step] = {n:p.detach().cpu().clone() for n,p in params.items()}
        if step and d["macro"]>best:
            best,best_step = d["macro"],step
        stopped = bool(step and step-best_step>=track.CHECK*track.PATIENCE)
        if step and checkpoint is not None:
            checkpoint(dict(step=step,weights=snapshots[step],optimizer=opt.state_dict(),
                            rng=torch.random.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state(tx.device) if tx.is_cuda else None,
                            snapshots=snapshots,best=float(best),best_step=best_step,
                            elapsed=elapsed,curve=json.dumps(curve),stopped=stopped))
        if stopped:
            break
    with torch.no_grad():
        for name,p in params.items():
            p.copy_(snapshots[best_step][name])
    return dict(best_step=best_step, curve=curve, train_seconds=elapsed, duration=perf_counter()-clock,
                missing_steps=[s for s in range(track.CHECK,track.STEPS+1,track.CHECK) if s not in snapshots]),snapshots


def arrays_to_bytes(arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer,**arrays)
    return buffer.getvalue()


def db_open():
    db = sqlite3.connect(DATABASE,timeout=180)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS runs (version TEXT, experiment TEXT, fold INTEGER, seed INTEGER, "
               "receipt TEXT, states BLOB, arrays BLOB, PRIMARY KEY(version,experiment,fold,seed))")
    db.execute("CREATE TABLE IF NOT EXISTS audits (source TEXT PRIMARY KEY, report TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS source_snapshots (path TEXT,sha256 TEXT,content BLOB,PRIMARY KEY(path,sha256))")
    db.execute("CREATE TABLE IF NOT EXISTS fusion_replays (version TEXT,fold INTEGER,seed INTEGER,report TEXT,"
               "PRIMARY KEY(version,fold,seed))")
    db.execute("CREATE TABLE IF NOT EXISTS failures (version TEXT, experiment TEXT, fold INTEGER, "
               "seed INTEGER, reason TEXT, recorded_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    db.commit()
    return db


def labels_for(roles):
    names = set(roles["fit"]+roles["select"])
    manifest = pd.read_csv(track.MANIFEST,usecols=["record_id","input_start_sec","input_end_sec",*TARGETS])
    result = {}
    for name, part in manifest[manifest.record_id.isin(names)].groupby("record_id"):
        ends = part.input_end_sec.to_numpy()
        np.testing.assert_array_equal(ends-60,part.input_start_sec.to_numpy())
        np.testing.assert_array_equal(ends,np.arange(60,60+len(ends)))
        result[name] = (part[TARGETS].to_numpy(),ends-1)
    return result


@torch.no_grad()
def score(model, data, nights):
    clock = perf_counter()
    rows, arrays = [], {}
    for person, (x, blocks) in nights.items():
        started = perf_counter()
        p = np.concatenate([predict(model,x[section],history) for section,history in blocks])
        if not np.isfinite(p).all() or not np.allclose(p.sum(1),1.,atol=2e-5):
            raise FloatingPointError(f"Invalid probabilities for {person}")
        y = data.labels[person][:,-1]
        inference = perf_counter()-started
        d,_,_ = event_score(y,p)
        d.update(person=person,diagnosis=diagnosis(y,p),inference_s=inference,score_s=perf_counter()-started-inference)
        rows.append(d)
        arrays[person], arrays[f"{person}_second"] = p,data.ends[person]-1
    return rows,arrays,perf_counter()-clock
