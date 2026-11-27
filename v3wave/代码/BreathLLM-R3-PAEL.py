"""轨道 A 的 PAEL 隐藏状态探针与独立 FP32 T03 基准。"""

import argparse
import importlib
import io
import json
import signal
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

r2 = importlib.import_module('BreathLLM-R2')
r1 = importlib.import_module('BreathLLM-R1')
import 发布复现 as repro

audit, track, b07 = r2.audit, r2.track, r2.b07
OUT = repro.OUTPUT/'BreathLLM-R3-PAEL结果.json'
CACHE = repro.OUTPUT/'PAEL缓存'
POSITIONS = (448,456,464,472,480)
FOLD, SEED, BATCH = 1, 4121, 32
ARMS = ('P01','P02','P03','P04','P05','P06')
UPPER = np.triu_indices(10,1)
OPT = dict(lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)


def cache_file(path, make, validate):
    if not path.exists():
        item = make()
        validate(item)
        repro.atomic(path,lambda stream:torch.save(item,stream))
    saved = torch.load(path,weights_only=True)
    validate(saved)
    return saved


def state_hash(model):
    import hashlib
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def fp32_reference():
    torch.set_float32_matmul_precision('high')
    ref, identity, replay = repro.reference()
    prior_path = repro.OUTPUT/'BreathLLM-R1结果.json'
    prior = json.loads(prior_path.read_text())
    assert prior['fp32']['pass'] and identity==prior['identity']['T03']
    ref.model.float().eval()
    ref.model.base.backbone.fixed_prefix = ref.model.base.backbone.fixed_prefix.float()
    # The archived trainable flags preserve R1 eval dispatch. No T03 forward has autograd.
    for value in ref.params.values():
        value.requires_grad_(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    return ref,identity,replay,track.hash_file(prior_path)


@torch.inference_mode()
def encode(ref,x,positions):
    captured = []
    handle = ref.model.base.decoder.register_forward_pre_hook(lambda _,args:captured.append(torch.cat(args,-1)))
    try:
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            output = ref.model(x)
    finally:
        handle.remove()
    assert len(captured)==1
    keep = slice(-len(positions),None)
    result = tuple(z[:,keep].contiguous().cpu() for z in (captured[0],*output.conditional_logits,output.log_prob))
    assert all(t.dtype==torch.float32 and t.isfinite().all() for t in result)
    return result


def model_cache(ref,role,schedule,positions,name,where,limit):
    output = []
    for start in range(0,len(schedule),1024):
        limit()
        inds = schedule[start:start+1024]
        def make():
            chunks,labels = [],[]
            for batch in np.array_split(inds,np.arange(BATCH,len(inds),BATCH)):
                limit()
                x,y = ref.batch(role,batch)
                chunks.append(encode(ref,x,positions))
                labels.append(y[:,-len(positions):].cpu())
            return dict(index=torch.from_numpy(inds),y=torch.cat(labels),
                        data=tuple(map(torch.cat,zip(*chunks,strict=True))))
        def validate(block):
            np.testing.assert_array_equal(block['index'].numpy(),inds)
            expected = np.empty((len(inds),len(positions)),dtype=np.int64)
            for i in np.unique(inds[:,0]):
                use = inds[:,0]==i
                person = ref.data.roles[role][i]
                expected[use] = ref.data.labels[person][inds[use,1],-len(positions):]
            np.testing.assert_array_equal(block['y'].numpy(),expected)
            assert all(t.shape[:2]==expected.shape and t.dtype==torch.float32 and t.isfinite().all() for t in block['data'])
            _,a,b,q,lp = block['data']
            torch.testing.assert_close(track.five_class_log_probs(a,b,q),lp,rtol=0,atol=2e-6)
        output.append(cache_file(where/f'{name}-{start:06}.pt',make,validate))
        if (start//1024)%8==7 or start+1024>=len(schedule):
            print(f'{name} FP32 {min(start+1024,len(schedule))}/{len(schedule)}',flush=True)
    return tuple(map(torch.cat,zip(*(r['data'] for r in output),strict=True))),torch.cat([r['y'] for r in output])


def signal_inputs(ref,limit):
    raw,info = {},{}
    for person in ref.data.roles['fit']+ref.data.roles['select']:
        limit()
        path = repro.DATA/f'{person}.rec'
        with audit.experiment_data.pyedflib.EdfReader(str(path)) as rec:
            positions = {n.casefold():i for i,n in enumerate(rec.getSignalLabels())}
            ids = [positions[name.casefold()] for name in track.PSG]
            rates = [int(rec.getSampleFrequency(i)) for i in ids]
            assert rates==[8]*4 and rec.getPhysicalDimension(ids[-1]).strip()=='%'
            flow,oxygen = (rec.readSignal(i).astype(np.float64) for i in (ids[0],ids[-1]))
        assert len(flow)==len(oxygen)==ref.data.signals[person].shape[-1]
        valid_o = np.isfinite(oxygen)&(oxygen>0)&(oxygen<=100)
        info[person] = dict(rates=rates,oxygen_unit='%',flow_nonfinite=int((~np.isfinite(flow)).sum()),
                            oxygen_invalid=int((~valid_o).sum()),flow_zero=int((flow==0).sum()))
        raw[person] = (flow,np.where(valid_o,oxygen,np.nan))
    return raw,info


def window_samples(x,width):
    return np.lib.stride_tricks.sliding_window_view(x,width)[::8]


def amplitude(x):
    valid = np.isfinite(x).all(-1)
    return np.where(valid,np.diff(np.quantile(np.where(np.isfinite(x),x,0),(.1,.9),axis=-1),axis=0)[0],np.nan)


def evidence_tables(waves):
    flow,oxygen = waves
    amp = amplitude(window_samples(flow,40))
    def oxygen_table(width):
        blocks = window_samples(oxygen,width)
        return np.where(np.isfinite(blocks).all(-1),np.median(blocks,axis=-1),np.nan)
    return amp,oxygen_table(160),oxygen_table(40)


def extract(table,stop):
    amp,o20,o5 = table
    start = (np.asarray(stop)-440)//8
    parts = amp[start[...,None]+np.arange(0,20,5)]
    count = np.isfinite(parts).sum(-1)
    ordered = np.sort(parts,axis=-1)
    reference = np.where(count==4,(ordered[...,1]+ordered[...,2])*.5,
                         np.where(count==3,ordered[...,1],np.nan))
    observed = amp[start[...,None]+np.arange(41,51)]
    base,last,prior = o20[start],o5[start+50],o5[start+45]
    oxygen = np.stack((np.clip((base-last)/5,-1,1),np.clip((prior-last)/3,-1,1)),axis=-1)
    return reference,observed,oxygen


def persistence(s):
    return np.minimum(s[...,:-1],s[...,1:]).mean(-1),np.minimum(s[...,UPPER[0]],s[...,UPPER[1]]).mean(-1)


def suppression(reference,observed,scale):
    valid = np.isfinite(reference)&(reference>=.05*scale)&np.isfinite(observed).all(-1)
    values = np.clip(1-np.where(valid[...,None],observed,0)/np.where(valid,reference+1e-6*scale,1)[...,None],0,1)
    return values,valid


def evidence(parts,scale):
    ref,observed,oxygen = parts
    s,vf = suppression(ref,observed,scale)
    vo = np.isfinite(oxygen).all(-1)
    seq,unordered = persistence(s)
    values = np.concatenate((np.stack((s[...,-1],s.mean(-1),s.std(-1),seq,unordered),axis=-1),oxygen),axis=-1)
    valid = np.concatenate((np.broadcast_to(vf[...,None],values.shape[:-1]+(5,)),
                            np.broadcast_to(vo[...,None],values.shape[:-1]+(2,))),axis=-1)
    return torch.from_numpy(np.where(valid,values,0).astype(np.float32)),torch.from_numpy(valid)


def fit_norm(values,valid):
    count = valid.sum(0)
    mean = values.masked_fill(~valid,0).sum(0)/count.clamp_min(1)
    std = (((values-mean).square().masked_fill(~valid,0).sum(0)/count.clamp_min(1)).sqrt()
           .clamp_min(1e-6).masked_fill(count==0,1))
    return mean,std


def normalize(values,valid,norm):
    mean,std = norm
    return ((values-mean)/std).masked_fill(~valid,0)


def inputs(arm,h,data,stats):
    h = (h-stats['hidden_mean'])/stats['hidden_std']
    if arm in ('P01','P06'):
        aux = h.new_zeros(h.shape[:-1]+(8,))
    else:
        v,m = data
        v = normalize(v,m,(stats['mean'],stats['std']))
        slots = [0,1,2,4 if arm=='P02' else 3,5,6]
        aux = torch.cat((v[...,slots],m[...,[0,5]].float()),-1)
    return torch.cat((h,aux),-1)


class Residual(nn.Sequential):
    def __init__(self,width):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            super().__init__(nn.Linear(width,24),nn.GELU(),nn.Linear(24,1))
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)

    def forward(self,h):
        return super().forward(h).squeeze(-1)


def persistent_logit(z):
    return -.5*(torch.logsumexp(-z/.5,dim=-1)-np.log(5))


def loss_parts(z,b,q,y,arm):
    nll = F.nll_loss(track.five_class_log_probs(z,b,q).flatten(0,1),y.flatten(),ignore_index=-1)
    normal = (y==0).all(-1)
    abnormal = (y>0).all(-1)
    use = normal|abnormal
    aux = z.new_zeros(())
    if arm in ('P04','P05','P06'):
        g = z.mean(-1) if arm=='P04' else persistent_logit(z)
        aux = F.binary_cross_entropy_with_logits(g[use],abnormal[use].float(),reduction='sum')/len(y)
    return nll+.5*aux,nll,aux,dict(normal=int(normal.sum()),abnormal=int(abnormal.sum()),
                                   all_hyp=int((y==4).all(-1).sum()),mixed=int((~use).sum()),
                                   invalid=int((y<0).any(-1).sum()))


def save_json(path,data):
    repro.atomic(path,lambda stream:stream.write(json.dumps(data,ensure_ascii=False,allow_nan=False).encode()))


def prepare(limit, progress):
    ref,parents,replay,r1hash = fp32_reference()
    indices = ref.train[:16384]
    assert len(indices)==16384
    raw,signals = signal_inputs(ref,limit)
    identity = dict(source=parents,r1=r1hash,fp32='math SDPA / highest / TF32 off / MHA fastpath off',
                    frozen_state_sha256=state_hash(ref.model),fit_index=audit.digest(indices.tobytes()),
                    positions=POSITIONS,encoding_batch=BATCH,
                    feature_support='55s raw 8Hz, full-segment finite mask, linear percentiles',
                    clock={p:audit.digest(v.tobytes()) for p,v in ref.data.ends.items()})
    identity = json.loads(json.dumps(identity))
    key = audit.digest(json.dumps(identity,sort_keys=True).encode())[:24]
    root = CACHE/key
    root.mkdir(parents=True,exist_ok=True)
    progress.update(identity=identity,cache_key=key,nights={},native_signals=signals,replay=replay)
    save_json(OUT,progress)
    tests = engineering_tests(ref,raw,indices,limit)
    timings = {}
    started = monotonic()
    nights = {}
    with np.load(io.BytesIO(ref.arrays)) as parent:
        for i,person in enumerate(ref.data.roles['select']):
            clock = ref.data.ends[person]-1
            ix = np.column_stack((np.full(len(clock),i),np.arange(len(clock))))
            data,y = model_cache(ref,'select',ix,(480,),person,root,limit)
            nights[person] = (tuple(a[:,0] for a in data),y[:,0],clock)
            progress['nights'][person] = dict(seconds=len(y),fp32=score(y[:,0].numpy(),data[-1][:,0].exp().numpy(),clock),
                                             bf16=score(y[:,0].numpy(),parent[person],clock))
            audit.check_predictions(parent[person],parent[person+'_second'],clock)
            save_json(OUT,progress)
    timings['fp32_nights'] = monotonic()-started
    progress['baseline'] = {kind:aggregate([nr[kind] for nr in progress['nights'].values()]) for kind in ('fp32','bf16')}
    progress['pending'].remove('P00')
    save_json(OUT,progress)
    started = monotonic()
    training,y = model_cache(ref,'fit',indices,POSITIONS,'fit',root,limit)
    timings['fp32_fit'] = monotonic()-started
    progress['pending'].remove('fit_cache')
    save_json(OUT,progress)
    started = monotonic()
    tables = {p:evidence_tables(w) for p,w in raw.items()}
    all_parts = [np.empty((len(indices),5,*extra)) for extra in ((),(10,),(2,))]
    for i,p in enumerate(ref.data.roles['fit']):
        limit()
        use = indices[:,0]==i
        ends = ref.data.ends[p][indices[use,1]]
        for out,computed in zip(all_parts,extract(tables[p],(ends[:,None]-60)*8+POSITIONS),strict=True):
            out[use] = computed
    candidates = all_parts[0][(all_parts[0]>0)&np.isfinite(all_parts[0])&(y.numpy()>=0)]
    assert len(candidates)
    scale = float(np.median(candidates))
    fit_evidence = evidence(all_parts,scale)
    h = training[0][y>=0]
    mean,std = fit_norm(*(v[y>=0] for v in fit_evidence))
    stats = dict(mean=mean,std=std,hidden_mean=h.mean(0),hidden_std=h.std(0,unbiased=False).clamp_min(1e-6),flow_scale=scale)
    test_features = {p:evidence(extract(tables[p],(v[2]+1)*8),scale) for p,v in nights.items()}
    def feature_cache():
        return dict(fit=fit_evidence,nights=test_features,stats=stats,train_index=torch.as_tensor(indices),
                    clock={p:torch.as_tensor(v[2]) for p,v in nights.items()})
    def validate_features(saved):
        np.testing.assert_array_equal(saved['train_index'].numpy(),indices)
        for p in nights:
            np.testing.assert_array_equal(saved['clock'][p].numpy(),nights[p][2])
        for left,right in zip(saved['fit'],fit_evidence,strict=True):
            torch.testing.assert_close(left,right,rtol=0,atol=0)
        for p in nights:
            for left,right in zip(saved['nights'][p],test_features[p],strict=True):
                torch.testing.assert_close(left,right,rtol=0,atol=0)
        for k,v in stats.items():
            torch.testing.assert_close(saved['stats'][k],v,rtol=0,atol=0)
    frozen_features = cache_file(root/'features.pt',feature_cache,validate_features)
    fit_evidence,test_features,stats = (frozen_features[k] for k in ('fit','nights','stats'))
    if progress['target_steps']==512:
        progress['continuation'] = verify_continuation(root,identity,indices,stats)
        progress['flow_diagnostics'] = flow_diagnostics(raw,tables,all_parts,y,nights,scale,test_features)
    tests.update(training_tests(training,y,fit_evidence,stats))
    timings['features_and_training_tests'] = monotonic()-started
    progress['pending'].remove('feature_cache')
    progress.update(tests=tests,seconds_by_phase=timings,
                    feature_valid_fit=(fit_evidence[1]& (y>=0)[...,None]).sum((0,1)).tolist(),
                    feature_valid_nights={p:m.sum(0).tolist() for p,(_,m) in test_features.items()},
                    flow_scale=scale,status='prepared')
    save_json(OUT,progress)
    return ref,root,training,y,fit_evidence,nights,test_features,stats


def math_tests():
    fixed = np.full((2,10),.6)
    np.testing.assert_allclose(persistence(fixed),.6,rtol=0,atol=1e-15)
    clustered = np.r_[np.ones(5),np.zeros(5)]
    alternating = clustered.reshape(2,5).T.ravel()
    seq, unordered = persistence(np.stack((clustered,alternating)))
    assert seq[0]>seq[1] and unordered[0]==unordered[1]
    assert amplitude(np.ones((1,40)))[0]==0
    assert np.isnan(amplitude(np.full((1,40),np.nan)))[0]
    wave = np.sin(np.arange(480)*2*np.pi/40)
    wave[360:] *= .3
    oxygen = np.r_[np.full(400,96),np.full(40,93),np.full(40,90)].astype(float)
    stop = np.array(POSITIONS)
    raw = extract(evidence_tables((wave,oxygen)),stop)
    multiplied = extract(evidence_tables((wave*4,oxygen)),stop)
    original,valid = evidence(raw,2.)
    changed,valid2 = evidence(multiplied,8.)
    torch.testing.assert_close(original,changed,rtol=0,atol=0)
    assert torch.equal(valid,valid2) and (original[-1,5:]>0).all()
    bad,m = evidence(extract(evidence_tables((wave*np.nan,oxygen*np.nan)),stop),2.)
    assert not bad.any() and not m.any() and bad.isfinite().all()
    for i,anchor in enumerate(stop):
        for left,right in ((0,anchor-440),(anchor,480)):
            f,o = wave.copy(),oxygen.copy()
            f[left:right]=np.nan
            o[left:right]=50.
            variant,m2 = evidence(extract(evidence_tables((f,o)),stop),2.)
            torch.testing.assert_close(original[i],variant[i],rtol=0,atol=0)
            assert torch.equal(valid[i],m2[i])
    z = torch.tensor([[1e4,-1e4,3.,4.,10.],[2.,2.,2.,2.,2.],[1.,2.,3.,4.,5.]])
    g = persistent_logit(z)
    assert g.isfinite().all() and (g>=z.min(-1).values).all() and (g<=z.mean(-1)).all()
    assert (g<=z.min(-1).values+.5*np.log(5)).all() and g[1]==2.
    for i in range(5):
        modified = z.clone()
        modified[:,i]+=1
        assert (persistent_logit(modified)>=g).all()
    y = torch.tensor([[0,1,0,1,0],[0,4,4,4,4],[-1,0,0,0,0]])
    total,_,aux,_ = loss_parts(z,torch.zeros_like(z),torch.zeros((*z.shape,3)),y,'P05')
    assert total.isfinite() and aux==0
    return ['幅值缩放与真实平坦','无序与有序持续性','缺失掩码与血氧方向','55秒支持与未来扰动','辅助logit范围、单调性与空监督']


def engineering_tests(ref,raw,indices,limit):
    tests = math_tests()
    x,_ = ref.batch('fit',indices[:32])
    case_a = encode(ref,x,POSITIONS)
    case_b = tuple(map(torch.cat,zip(*(encode(ref,row,POSITIONS) for row in x[:4].split(1)),strict=True)))
    comparison = r1.r0.compare_layers({'probability':case_a[-1][:4].exp().numpy(),'log probability':case_a[-1][:4].numpy()},
                                      {'probability':case_b[-1].exp().numpy(),'log probability':case_b[-1].numpy()})
    assert comparison['layers']['probability']['max_abs']<=1e-4
    assert comparison['layers']['log probability']['max_abs']<=1e-3
    assert comparison['decision_flips']==comparison['alarm_flips']==0
    rng = np.random.default_rng(74121)
    person = ref.data.roles['fit'][indices[0,0]]
    rates,native = audit.read_native_prefix(repro.DATA/f'{person}.rec',61)
    baseline = audit.preprocess_prefix(rates,native)
    np.testing.assert_array_equal(baseline[0],raw[person][0][:488].astype(np.float32))
    valid = np.isfinite(baseline[-1])
    np.testing.assert_array_equal(baseline[-1,valid],raw[person][1][:488][valid].astype(np.float32))
    full = audit.standardize(baseline,ref.data.mean,ref.data.scale)
    np.testing.assert_array_equal(full,ref.data.signals[person][:,:488])
    wave = torch.as_tensor(full[None,:,:480],device=ref.device)
    original = encode(ref,wave,POSITIONS)
    for i,stop in enumerate(POSITIONS):
        limit()
        changed = audit.perturb_native(native,rates,stop//8,'noise',rng)
        variant = audit.preprocess_prefix(rates,changed)
        np.testing.assert_array_equal(baseline[:,:stop],variant[:,:stop])
        frozen = encode(ref,torch.as_tensor(audit.standardize(variant,ref.data.mean,ref.data.scale)[None,:,:480],
                                           device=ref.device),POSITIONS)
        for a,b in zip(original,frozen,strict=True):
            torch.testing.assert_close(a[:,:i+1],b[:,:i+1],rtol=0,atol=1e-6)
    return dict(math=tests,fp32_batch_check=comparison,native_causality=dict(person=person,rates=rates,positions=POSITIONS))


def training_tests(training,y,evidence,stats):
    h,a,b,q,lp = (t[:32] for t in training)
    model = Residual(h.shape[-1]+8)
    x = inputs('P05',h,tuple(t[:32] for t in evidence),stats)
    z = a+model(x)
    torch.testing.assert_close(track.five_class_log_probs(z,b,q),lp,rtol=0,atol=2e-6)
    saved_b,saved_q = b.clone(),q.clone()
    loss,*_ = loss_parts(z,b,q,y[:32],'P05')
    loss.backward()
    assert model[-1].weight.grad.abs().max()>0 and all(not t.requires_grad for t in training)
    torch.testing.assert_close(b,saved_b,rtol=0,atol=0)
    torch.testing.assert_close(q,saved_q,rtol=0,atol=0)
    return dict(zero_initialization=True,only_residual_gradients=True,bq_immutable=True,fit_only_statistics=True,
                parameter_count=sum(t.numel() for t in model.parameters()))


def score(y,p,clock):
    audit.check_predictions(p,clock,clock)
    from sklearn.metrics import roc_auc_score, average_precision_score
    counts,positive,match = b07.event_score(y,p)
    audit.check_matches(y,positive,match)
    alarms = b07.runs(positive.astype(int),gap=b07.OPTIONS['gap'],min_positive=b07.OPTIONS['min_run'])
    sums = r2.interval_sums(p[:,1:],alarms,positive)
    alarms[:,2] = sums.argmax(1)+1
    hit = np.isfinite(match[:,3])
    ids = np.searchsorted(alarms[:,0],(match[hit,0]+match[hit,4]).astype(int))
    matched_by = dict(zip(ids,np.flatnonzero(hit),strict=True))
    a = 1-p[:,0].astype(np.float64)
    observed = y>=0
    truth = y[observed]>0
    prob = a[observed]
    safe = np.clip(prob,np.finfo(float).eps,1-np.finfo(float).eps)
    abnormal_error = np.square(prob-truth)
    hypnormal = (y==4)|(y==0)
    hypclass = y[hypnormal]==4
    hyps = []
    full_events = np.asarray(match[:,:3],dtype=int)
    for j,(start,end,_) in enumerate(full_events):
        if y[start]!=4:
            continue
        matched_alarm = next((alarm_id for alarm_id,event_id in matched_by.items() if event_id==j),None)
        hyps.append(dict(index=j,start=int(clock[0]+start),stop=int(clock[0]+end),matched=bool(hit[j]),
                         prediction=int(match[j,3]) if hit[j] else 0,
                         **r2.detection_features(start,end,a,positive,alarms,matched_alarm,int(clock[0]))))
    return dict(counts=counts,cm=track.confusion(y,p.argmax(1)).tolist(),normal=int((y==0).sum()),
                normal_fp=int((positive&(y==0)).sum()),brier_sum=float(abnormal_error.sum()),
                nll_sum=float(np.where(truth,-np.log(safe),-np.log1p(-safe)).sum()),
                hyp_a_auc=float(roc_auc_score(hypclass,a[hypnormal])) if hypclass.any() and (~hypclass).any() else None,
                hyp_a_ap=float(average_precision_score(hypclass,a[hypnormal])) if hypclass.any() and (~hypclass).any() else None,
                alarms=[list(map(int,(*bounds,int(j in matched_by),matched_by.get(j,-1)))) for j,bounds in enumerate(alarms+np.array([clock[0],clock[0],0]))],
                references=(full_events+np.array([clock[0],clock[0],0])).tolist(),hyp_events=hyps)


def aggregate(rows):
    cm = np.sum([r['counts']['event_cm'] for r in rows],axis=0)
    pred = np.sum([r['counts']['alarms_by_cls'] for r in rows],axis=0)
    denom = cm.sum()+pred.sum()
    truth = cm.sum(1)
    tp = cm[:,1:].diagonal()
    hits = cm[:,1:].sum(1)
    hours = sum(r['counts']['seconds'] for r in rows)/3600
    classes = {name:dict(tp=int(t),truth=int(n),pred=int(p),precision=float(t/max(p,1)),
                         recall=float(t/max(n,1)),f1=float(2*t/max(n+p,1)),detected=int(d))
               for name,t,n,p,d in zip(b07.CLASS_NAMES[1:],tp,truth,pred,hits,strict=True)}
    return dict(hyp_recall=classes['Hypopnea']['recall'],hyp_precision=classes['Hypopnea']['precision'],
                hyp_f1=classes['Hypopnea']['f1'],hyp_detection=hits[-1]/max(truth[-1],1),
                apnea_detection=hits[:3].sum()/max(truth[:3].sum(),1),detect_f1=2*hits.sum()/denom,
                typed_f1=2*tp.sum()/denom,fa_h=(pred.sum()-hits.sum())/hours,
                macro=track.macro_f1(np.sum([r['cm'] for r in rows],axis=0)),classes=classes,event_cm=cm.tolist(),
                normal_fpr=sum(r['normal_fp'] for r in rows)/max(sum(r['normal'] for r in rows),1),
                a_brier=sum(r['brier_sum'] for r in rows)/(hours*3600),a_nll=sum(r['nll_sum'] for r in rows)/(hours*3600),
                hyp_a_subject_equal_auc=r2.mean_defined([r['hyp_a_auc'] for r in rows]),
                hyp_a_subject_equal_ap=r2.mean_defined([r['hyp_a_ap'] for r in rows]))


def aligned_alarms(before,after):
    old = np.asarray(before,dtype=int).reshape(-1,5)
    new = np.asarray(after,dtype=int).reshape(-1,5)
    i = np.lexsort((np.arange(len(old)),old[:,0]))
    j = np.lexsort((np.arange(len(new)),new[:,0]))
    a,b = old[i],new[j]
    overlap = (np.minimum(a[:,None,1],b[None,:,1])-np.maximum(a[:,None,0],b[None,:,0])).clip(0)
    # Fractional weights break equal overlap totals by sorted starts and row order.
    if not overlap.size:
        return []
    priority = (len(a)-np.arange(len(a)))[:,None]*(len(b)-np.arange(len(b)))[None,:]
    weights = overlap + (overlap>0)*priority/(2.*priority.max()*max(1,min(len(a),len(b))))
    row,col = linear_sum_assignment(weights,maximize=True)
    return [(int(i[m]),int(j[n]),int(overlap[m,n])) for m,n in zip(row,col,strict=True) if overlap[m,n]>0]


def transitions(base,current):
    alignment = aligned_alarms(base['alarms'],current['alarms'])
    aligned_old = {i for i,_,_ in alignment}
    aligned_new = {i for _,i,_ in alignment}
    refs = np.asarray(current['references'],dtype=int)
    np.testing.assert_array_equal(refs,base['references'])
    def new_alarm(alarms,other):
        records = []
        for j,(start,end,cls,match,ref_index) in enumerate(alarms):
            if j in other:
                continue
            overlap = np.maximum(0,np.minimum(end,refs[:,1])-np.maximum(start,refs[:,0])) if len(refs) else []
            distance = np.maximum(0,np.maximum(start-refs[:,1],refs[:,0]-end)) if len(refs) else []
            closest = int(np.lexsort((np.arange(len(refs)),refs[:,0],-overlap,distance))[0]) if len(refs) else None
            records.append(dict(index=j,start=start,stop=end,duration=end-start,prediction=cls,
                                matched=bool(match),correct_subtype=bool(match and cls==refs[ref_index,2]),
                                matched_reference=ref_index if match else None,
                                nearest_reference=refs[closest].tolist() if closest is not None else None,
                                overlap_seconds=int(overlap[closest]) if closest is not None else 0,
                                distance_seconds=int(distance[closest]) if closest is not None else None))
        return records
    old_by = {r['index']:r for r in base['hyp_events']}
    new_by = {r['index']:r for r in current['hyp_events']}
    assert old_by.keys()==new_by.keys()
    return dict(alignment=alignment,added=new_alarm(current['alarms'],aligned_new),
                lost=new_alarm(base['alarms'],aligned_old),
                rescued_hyp=[dict(before=old_by[k],after=v) for k,v in new_by.items() if v['matched'] and not old_by[k]['matched']],
                lost_hyp=[dict(before=old_by[k],after=v) for k,v in new_by.items() if not v['matched'] and old_by[k]['matched']],
                fewer_than_five_positive=sum(v['production_positive_seconds']<5 for v in new_by.values()))


def train_arm(arm,steps,directory,training,y,features,stats,limit):
    h,a,b,q,lp = training
    model = Residual(h.shape[-1]+8)
    initialization = state_hash(model)
    opt = torch.optim.AdamW(model.parameters(),**OPT)
    resume = directory/f'{arm}-fit.pt'
    requested = directory/f'{arm}-{steps}.pt'
    path = requested if requested.exists() else resume
    losses = []
    if path.exists():
        state = torch.load(path,weights_only=True)
        assert state['initialization']==initialization and len(state['losses'])<=steps
        model.load_state_dict(state['weights'])
        opt.load_state_dict(state['optimizer'])
        torch.random.set_rng_state(state['rng'])
        assert state['losses'][-1]['step']==len(state['losses'])
        assert torch.equal(state['rng'],torch.random.get_rng_state())
        assert all(torch.equal(value,state['weights'][name]) for name,value in model.state_dict().items())
        current = opt.state_dict()
        assert current['param_groups']==state['optimizer']['param_groups']
        assert all(torch.equal(value,state['optimizer']['state'][name][slot])
                   for name,buffers in current['state'].items() for slot,value in buffers.items())
        losses = state['losses']
    for i in range(len(losses),steps):
        limit()
        j = slice(i*BATCH,(i+1)*BATCH)
        x = inputs(arm,h[j],tuple(t[j] for t in features),stats)
        z = a[j]+model(x)
        total,nll,aux,block = loss_parts(z,b[j],q[j],y[j],arm)
        assert total.isfinite()
        opt.zero_grad(set_to_none=True)
        total.backward()
        assert all(v.grad is not None and v.grad.isfinite().all() for v in model.parameters())
        opt.step()
        losses.append(dict(step=i+1,total=total.item(),nll=nll.item(),aux=aux.item(),blocks=block))
        state = dict(weights=model.state_dict(),optimizer=opt.state_dict(),rng=torch.random.get_rng_state(),
                     initialization=initialization,losses=losses)
        if (i+1)%32==0:
            repro.atomic(resume,lambda stream:torch.save(state,stream))
        if i+1 in (128,256,384,512):
            repro.atomic(directory/f'{arm}-{i+1}.pt',lambda stream:torch.save(state,stream))
    return model.eval(),dict(losses=losses,initialization=initialization,parameters=sum(t.numel() for t in model.parameters()),
                             input_slots=h.shape[-1]+8,physiology_slots=0 if arm in ('P01','P06') else 8,
                             final_hash=state_hash(model))


def bootstrap_metrics(rows,draws):
    def boot(counts):
        return np.asarray(counts)[draws].sum(1)
    confusion = boot([r['counts']['event_cm'] for r in rows])
    alarms = boot([r['counts']['predicted'] for r in rows])
    typ = confusion[:,np.arange(4),np.arange(1,5)]
    total = confusion.sum((1,2))
    matched = confusion[:,:,1:].sum((1,2))
    seconds = boot([r['counts']['seconds'] for r in rows])
    sec_cm = boot([r['cm'] for r in rows])
    den = sec_cm.sum(1)+sec_cm.sum(2)
    with np.errstate(invalid='ignore',divide='ignore'):
        return dict(hyp_recall=typ[:,-1]/confusion[:,-1,:].sum(1),
                    hyp_detection=confusion[:,-1,1:].sum(1)/confusion[:,-1,:].sum(1),
                    apnea_detection=confusion[:,:3,1:].sum((1,2))/confusion[:,:3,:].sum((1,2)),
                    detect_f1=2*matched/(total+alarms),typed_f1=2*typ.sum(1)/(total+alarms),
                    fa_h=3600*(alarms-matched)/seconds,
                    macro=np.divide(2*sec_cm.diagonal(axis1=1,axis2=2),den,
                                    out=np.zeros_like(den,dtype=float),where=den>0).mean(1))


def hypotheses(results):
    names = list(results['P00']['subjects'])
    rng = np.random.default_rng(74121)
    draws = rng.integers(len(names),size=(10000,len(names)))
    samples = {k:bootstrap_metrics([v['subjects'][n] for n in names],draws) for k,v in results.items()}
    contrasts = {}
    for arm,parent in [('P02','P01'),('P03','P02'),('P05','P04'),('P05','P03'),('P05','P06'),('P04','P03'),
                       *((arm,'P00') for arm in ARMS)]:
        high,low = results[arm],results[parent]
        metrics = {k:dict(delta=float(high['pooled'][k]-low['pooled'][k]),**r2.interval(samples[arm][k]-samples[parent][k]))
                   for k in samples[arm]}
        delta = {k:v['delta'] for k,v in metrics.items()}
        net = {}
        for name in names:
            a,b = high['subjects'][name]['hyp_events'],low['subjects'][name]['hyp_events']
            assert len(a)==len(b)
            net[name] = sum(r['prediction']==4 for r in a)-sum(r['prediction']==4 for r in b)
        people = sum(len(high['subjects'][n]['hyp_events'])>0 for n in names)
        after_top = sum(net.values())-max(net.values())
        contrasts[f'{arm}-{parent}'] = dict(metrics=metrics,people=people,recall_improved=sum(d>0 for d in net.values()),
                                            correct_hyp_delta=net,correct_hyp_delta_excluding_top=after_top,
                                            mechanism=delta['hyp_recall']>=.02 and delta['fa_h']<=.10 and delta['typed_f1']>=0,
                                            advances=(delta['hyp_recall']>=.05 and delta['detect_f1']>=.01 and delta['typed_f1']>=.01
                                                      and delta['fa_h']<=.2 and delta['apnea_detection']>=-.01 and delta['macro']>=-.005
                                                      and people>=4 and sum(d>0 for d in net.values())>=4 and after_top>0))
    return contrasts


@torch.inference_mode()
def predict_arm(model,arm,p0,features,stats):
    h,a,b,q,lp = p0
    sections = []
    for start in range(0,len(h),4096):
        j = slice(start,start+4096)
        x = inputs(arm,h[j],tuple(v[j] for v in features),stats)
        sections.append(track.five_class_log_probs(a[j]+model(x),b[j],q[j]).exp())
    return torch.cat(sections).numpy()


def run_experiment(prepared,steps,limit,progress):
    ref,directory,training,y,features,nights,test_features,stats = prepared
    ref.model.requires_grad_(False)
    before = state_hash(ref.model)
    assert before==progress['identity']['frozen_state_sha256']
    base = {p:r['fp32'] for p,r in progress['nights'].items()}
    results = {'P00':dict(subjects=base,pooled=aggregate(list(base.values())))}
    progress['groups'] = results
    baseline_outputs = {p:night[-1].exp().numpy() for p,(night,_,_) in nights.items()}
    progress['pending'] = list(ARMS)
    progress['optimizer'] = OPT
    for arm in ARMS:
        limit()
        started = monotonic()
        model,fit = train_arm(arm,steps,directory,training,y,features,stats,limit)
        fit['seconds'] = monotonic()-started
        subjects,changes,outputs = {},{},{}
        started = monotonic()
        for person,(night,target,clock) in nights.items():
            limit()
            pred = predict_arm(model,arm,night,test_features[person],stats)
            subjects[person] = score(target.numpy(),pred,clock)
            changes[person] = transitions(base[person],subjects[person])
            outputs[person] = pred
            changes[person]['hyp_recall_delta'] = (subjects[person]['counts']['event_cm'][3][4]-base[person]['counts']['event_cm'][3][4])/max(len(base[person]['hyp_events']),1)
            changes[person]['apnea_detection_delta'] = float(aggregate([subjects[person]])['apnea_detection']-aggregate([base[person]])['apnea_detection'])
            if arm=='P05':
                for item in changes[person]['rescued_hyp']:
                    start,stop = item['before']['start']-clock[0],item['before']['stop']-clock[0]
                    item['a_scores'] = {k:(1-v[start:stop,0].astype(np.float64)).tolist()
                                        for k,v in (('P00',baseline_outputs[person]),('P05',pred))}
        results[arm] = dict(fit=fit,subjects=subjects,pooled=aggregate(list(subjects.values())),
                            transitions=changes,scoring_seconds=monotonic()-started,
                            transition_counts=dict(rescued_hyp=sum(len(v['rescued_hyp']) for v in changes.values()),
                                lost_hyp=sum(len(v['lost_hyp']) for v in changes.values()),
                                new_alarms=sum(len(v['added']) for v in changes.values()),
                                new_matched=sum(sum(r['matched'] for r in v['added']) for v in changes.values()),
                                new_false_alarms=sum(sum(not r['matched'] for r in v['added']) for v in changes.values()),
                                removed_false_alarms=sum(sum(not r['matched'] for r in v['lost']) for v in changes.values())))
        results[arm]['transition_counts']['net_matched_hyp'] = (results[arm]['transition_counts']['rescued_hyp']-
                                                               results[arm]['transition_counts']['lost_hyp'])
        if steps>128:
            results[arm]['offline_operating_point'] = operating_point(nights,baseline_outputs,outputs)
            results[arm]['checkpoint_sha256'] = track.hash_file(directory/f'{arm}-{steps}.pt')
        progress['pending'].remove(arm)
        save_json(OUT,progress)
        print(f'{arm} 完成 {steps} 步和整夜评分',flush=True)
    assert len({r['fit']['initialization'] for k,r in results.items() if k!='P00'})==1
    assert before==state_hash(ref.model) and all(p.grad is None and not p.requires_grad for p in ref.model.parameters())
    progress['bootstrap'] = dict(draws=10000,seed=74121,unit='subject')
    progress['hypotheses'] = hypotheses(results)
    progress['status'] = f'completed_{steps}'
    if steps>128:
        progress['sealed_milestones'] = [s for s in (128,256,384,512) if s<=steps]
    save_json(directory/f'result-{steps}.json',progress)


def verify_continuation(directory,identity,indices,stats):
    first = json.loads((directory/'result-128.json').read_text())
    assert first['status']=='completed_128' and first['identity']==identity
    archived = torch.load(directory/'features.pt',weights_only=True)
    np.testing.assert_array_equal(archived['train_index'].numpy(),indices)
    for name,value in stats.items():
        torch.testing.assert_close(value,archived['stats'][name],rtol=0,atol=0)
    record = {}
    for arm in ARMS:
        path = directory/f'{arm}-128.pt'
        saved = torch.load(path,weights_only=True)
        reported = first['groups'][arm]['fit']
        model = Residual(len(stats['hidden_mean'])+8)
        assert state_hash(model)==saved['initialization']==reported['initialization']
        model.load_state_dict(saved['weights'])
        assert state_hash(model)==reported['final_hash']
        assert saved['losses']==reported['losses'] and len(saved['losses'])==128
        assert [step['step'] for step in saved['losses']]==list(range(1,129))
        assert len(saved['optimizer']['state'])==len(list(model.parameters()))
        for moment in saved['optimizer']['state'].values():
            assert moment['step'].item()==128
            assert moment['exp_avg'].isfinite().all() and moment['exp_avg_sq'].isfinite().all()
        assert saved['rng'].dtype==torch.uint8 and saved['rng'].shape==torch.random.get_rng_state().shape
        live_path = directory/f'{arm}-fit.pt'
        live = torch.load(live_path,weights_only=True)
        assert live['losses'][:128]==saved['losses']
        if len(live['losses'])==128:
            assert track.hash_file(live_path)==track.hash_file(path)
        record[arm] = dict(first_sha256=track.hash_file(path),resume_step=len(live['losses']),
                           training_cursor=len(live['losses'])*BATCH)
    return dict(records=record,train_index=identity['fit_index'],norm_sha256=track.hash_file(directory/'features.pt'),
                cache_key=directory.name)


def value_distribution(values):
    return dict(n=int(len(values)),quantiles=np.quantile(values,[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist() if len(values) else None)


def flow_diagnostics(raw,tables,fit_parts,fit_labels,nights,scale,test_features):
    from scipy.stats import spearmanr
    def series(parts,y):
        values,valid = suppression(parts[0],parts[1],scale)
        seq,unordered = persistence(values)
        tv = np.abs(np.diff(values,axis=-1)).sum(-1)
        rebuilt = (2*values.sum(-1)-values[...,0]-values[...,-1]-tv)/18
        max_error = float(np.abs(seq[valid]-rebuilt[valid]).max(initial=0))
        assert max_error<=1e-6
        return np.stack((seq,unordered,values.mean(-1),seq-unordered,tv),axis=-1)[valid],y[valid],max_error
    def summarize(values,labels):
        data = {}
        cohorts = {'all':np.ones(len(labels),dtype=bool),'normal':labels==0,'hypopnea':labels==4,
                   'apnea':(labels>0)&(labels<4),'invalid_label':labels<0}
        for tag,use in cohorts.items():
            x = values[use]
            corr = {}
            for j,name in ((1,'unordered'),(2,'mean')):
                active = len(x)>1 and np.ptp(x[:,0])>0 and np.ptp(x[:,j])>0
                corr[name] = dict(pearson=float(np.corrcoef(x[:,0],x[:,j])[0,1]) if active else None,
                                  spearman=float(spearmanr(x[:,0],x[:,j]).statistic) if active else None)
            data[tag] = dict(correlations=corr,distributions={name:value_distribution(x[:,i])
                for i,name in enumerate(('J_seq','J_unordered','mean','J_difference','TV'))})
        return data
    values,y,fit_error = series(fit_parts,fit_labels.numpy())
    fitted = summarize(values,y)
    select,by_person = [],{}
    for person,(_,y,clock) in nights.items():
        stops = (clock+1)*8
        found = series(extract(tables[person],stops),y.numpy())
        select.append(found)
        by_person[person] = summarize(*found[:2])
    verification = summarize(np.concatenate([x[0] for x in select]),np.concatenate([x[1] for x in select]))
    person = min(test_features,key=lambda p:int(test_features[p][1][:,0].sum()))
    clock = nights[person][2]
    state = (clock+1)*8
    amp = tables[person][0]
    start = (state-440)//8
    reference,observed,_ = extract(tables[person],state)
    refs = amp[start[:,None]+np.arange(0,20,5)]
    used = np.r_[np.arange(0,160),np.arange(328,440)]
    nonfinite = (~np.isfinite(raw[person][0][start[:,None]*8+used])).any(1)
    _,valid = suppression(reference,observed,scale)
    np.testing.assert_array_equal(valid,test_features[person][1][:,0].numpy())
    insufficient = np.isfinite(refs).sum(1)<3
    low = np.isfinite(reference)&(reference<.05*scale)
    dyn = ~np.isfinite(observed).all(1)
    units = [nonfinite,insufficient,low,dyn]
    other = ~valid&~(insufficient|low|dyn)
    units.append(other)
    keys = ('nonfinite_raw','insufficient_reference_segments','reference_below_floor','invalid_dynamic_segment','other')
    codes = np.column_stack(units)
    pattern,counts = np.unique(codes[~valid],axis=0,return_counts=True)
    quiet = dict(person=person,samples=len(valid),valid=int(valid.sum()),invalid=int((~valid).sum()),
                 independent_raw_mask_available=False,raw_mask_invalid_count=None,
                 floor=.05*scale,overlapping={k:int(v.sum()) for k,v in zip(keys,units,strict=True)},
                 exclusive={k:int((codes[:,i]&(codes.sum(1)==1)).sum()) for i,k in enumerate(keys)},
                 combinations=[dict(causes=[k for k,on in zip(keys,row,strict=True) if on],windows=int(n))
                               for row,n in zip(pattern,counts,strict=True)],
                 raw_flow=value_distribution(raw[person][0][np.isfinite(raw[person][0])]),
                 raw_5s_amplitude=value_distribution(amp[np.isfinite(amp)]),
                 reference_amplitude=value_distribution(reference[np.isfinite(reference)]),
                 dynamic_amplitude=value_distribution(observed[np.isfinite(observed)]))
    return dict(max_identity_error=dict(fit=fit_error,select=max(x[2] for x in select)),
                fitted=fitted,validation=verification,validation_subjects=by_person,
                limited_flow_subject=quiet)


def operating_point(nights,baseline,outputs):
    from sklearn.metrics import roc_auc_score, average_precision_score
    def get(paths):
        a = {p:pred[:,1:].sum(1) for p,pred in paths.items()}
        normal = np.concatenate([a[p][y.numpy()==0] for p,(_,y,_) in nights.items()])
        hyp = np.concatenate([a[p][y.numpy()==4] for p,(_,y,_) in nights.items()])
        values = np.r_[normal,hyp]
        labels = np.repeat([False,True],[len(normal),len(hyp)])
        return a,normal,hyp,float(roc_auc_score(labels,values)),float(average_precision_score(labels,values))
    base,base_normal,base_hyp,base_auc,base_ap = get(baseline)
    scores,normal,hyp,auc,ap = get(outputs)
    target_count = int((base_normal>=.5).sum())
    assert 0<target_count<len(normal) and len(normal)==len(base_normal) and len(hyp)==len(base_hyp)
    cutoff = np.partition(normal,len(normal)-target_count)[len(normal)-target_count]
    weight = (target_count-int((normal>cutoff).sum()))/int((normal==cutoff).sum())
    def count(v):
        return int((v>cutoff).sum())+weight*int((v==cutoff).sum())
    assert count(normal)==target_count
    return dict(unit='seconds',rule='pooled normal FPR matched with fractional cutoff ties',
                target_normal_fpr=target_count/len(normal),cutoff=float(cutoff),tie_weight=weight,
                baseline_hyp_sensitivity=float((base_hyp>=.5).mean()),
                current_fixed_hyp_sensitivity=float((hyp>=.5).mean()),
                matched_fpr_hyp_sensitivity=count(hyp)/len(hyp),
                change_at_matched_fpr=count(hyp)/len(hyp)-float((base_hyp>=.5).mean()),
                pooled_auc=dict(P00=base_auc,current=auc),pooled_auprc=dict(P00=base_ap,current=ap),
                subjects={p:dict(normal=int((y==0).sum()),hyp=int((y==4).sum()),
                                 matched_fpr_hyp_positive=count(scores[p][y.numpy()==4]),
                                 baseline_hyp_positive=int((base[p][y.numpy()==4]>=.5).sum()))
                          for p,(_,y,_) in nights.items()})


def run_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--started-at',type=float,default=monotonic())
    parser.add_argument('--steps',type=int,choices=(0,128,512),default=128)
    args = parser.parse_args()
    def limit():
        if monotonic()-args.started_at>=600:
            raise TimeoutError('达到 10 分钟软限')
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('达到 15 分钟硬限')))
    signal.alarm(max(1,int(900-(monotonic()-args.started_at))))
    torch.set_num_threads(2)
    begin = monotonic()
    previous = json.loads(OUT.read_text()) if OUT.exists() else {}
    progress = dict(status='preparing',fold=FOLD,seed=SEED,target_steps=args.steps,
                    source_sha256=track.hash_file(Path(__file__)),
                    pending=['P00','fit_cache','feature_cache',*ARMS])
    try:
        limit()
        prepared = prepare(limit,progress)
        if previous.get('cache_key')==progress.get('cache_key'):
            original = previous.get('initial_preparation_seconds',previous.get('seconds_by_phase'))
            if original is not None:
                progress['initial_preparation_seconds'] = original
        if args.steps:
            milestones = (256,384,512) if args.steps==512 else (args.steps,)
            root = prepared[1]
            for step in milestones:
                limit()
                sealed = root/f'result-{step}.json'
                if step>128 and sealed.exists():
                    archived = json.loads(sealed.read_text())
                    assert archived['identity']==progress['identity'] and archived['status']==f'completed_{step}'
                    for arm in ARMS:
                        assert archived['groups'][arm]['checkpoint_sha256']==track.hash_file(root/f'{arm}-{step}.pt')
                    progress.update(groups=archived['groups'],hypotheses=archived['hypotheses'],
                                    status=archived['status'],pending=[],sealed_milestones=archived['sealed_milestones'])
                else:
                    run_experiment(prepared,step,limit,progress)
        else:
            progress['status']='prepared'
    except TimeoutError as exc:
        progress.update(status='partial',reason=str(exc))
    except Exception as exc:
        progress.update(status='failed',reason=repr(exc))
        raise
    finally:
        signal.alarm(0)
        progress.update(seconds=monotonic()-args.started_at,process_seconds=monotonic()-begin)
        save_json(OUT,progress)
        print(progress['status'],progress['seconds'],flush=True)


if __name__=='__main__':
    run_main()
