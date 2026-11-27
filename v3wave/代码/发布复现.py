"""公开交接包的外部资产、T03 重建和归档校验。"""

import io
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pyedflib
import torch

import M3首批审计 as audit
from breathllm_paths import ASSETS, DATA, MANIFEST, OUTPUT, PROJECT, SPLIT, WEIGHTS
from src.data.sliding_window_60s import tail_labels, window_ends
from src.labels.resp import clock_seconds, read_events
from src.labels.second_level_labels import rasterize
from src.labels.ucddb_mapping import IGNORE, LABEL_VERSION

b07, track = audit.b07, audit.track
POSITIONS = (448, 456, 464, 472, 480)
OPT = dict(lr=2e-4, weight_decay=.01, betas=(.9,.999), eps=1e-8, amsgrad=False,
           maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)
HISTORY = 'B07基座改造结果.sqlite'
LOCAL = OUTPUT / "T03.sqlite"
DEVICE = "cuda"


def assets():
    return json.loads(ASSETS.read_text())


def roster():
    expected = assets()['dataset']['nested_split_sha256']
    if track.hash_file(SPLIT) != expected:
        raise ValueError("锁定划分 SHA256 不一致")
    return track.route(pd.read_csv(SPLIT), 1, (1, 2, 3))


def atomic(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def verify_files(files, hint):
    result = {}
    for path,info in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"缺少 {path}。{hint}")
        sha = track.hash_file(path)
        if path.stat().st_size != info['bytes'] or sha != info['sha256']:
            raise ValueError(f"外部资产大小或 SHA256 不符 {path}。{hint}")
        result[path.name] = sha
    return result


def require_data():
    pins = assets()['dataset']
    if not MANIFEST.exists():
        raise FileNotFoundError(f"缺少 {MANIFEST}。从 PhysioNet 下载 UCDDB 1.0.0 后运行 python breathllm.py prepare")
    if track.hash_file(MANIFEST) != pins['development_manifest_sha256']:
        raise ValueError("开发窗口清单 SHA256 不符，请运行 python breathllm.py prepare 重建")
    return verify_files({DATA/name: info for name,info in pins['files'].items()},
                        "按 README 下载 UCDDB 1.0.0 并设置 BREATHLLM_DATA_ROOT")


def weights():
    files = {WEIGHTS/name/info['file']: info for name,bundle in assets()['weights'].items()
             for info in bundle['files']}
    return verify_files(files,"按 README 下载锁定版本权重，运行 python breathllm.py weights")


def prepare():
    pins = assets()['dataset']
    records = sorted({person for r in roster().values() for person in r['select']})
    verify_files({DATA/name:info for name,info in pins['files'].items()},
                 "从 https://physionet.org/content/ucddb/1.0.0/ 下载开发记录及分期和呼吸标注")
    windows = []
    for person in records:
        with pyedflib.EdfReader(str(DATA/f'{person}.rec')) as edf:
            channels = {name.casefold():i for i,name in enumerate(edf.getSignalLabels())}
            duration = edf.getFileDuration()
            if duration != int(duration):
                raise ValueError(f'EDF 时长必须为整数秒 {person}')
            duration = int(duration)
            start = edf.getStartdatetime().time().isoformat(timespec='seconds')
            for name in track.PSG:
                i = channels[name.casefold()]
                rate = edf.getSampleFrequency(i)
                if rate % 8 or edf.getNSamples()[i] != rate*duration:
                    raise ValueError(f'通道采样率或时长不一致 {person}/{name}')
        stages = np.loadtxt(DATA/f'{person}_stage.txt',dtype=np.int16,ndmin=1)
        events = pd.DataFrame(read_events(DATA/f'{person}_respevt.txt',person,clock_seconds(start),duration,stages))
        if not events.in_record.all():
            raise ValueError(f'呼吸标注超出记录边界 {person}')
        events = events.sort_values(['start_sec','source_line'],kind='stable').reset_index(drop=True)
        labels = rasterize(duration,stages,events).labels
        ends = window_ends(duration)
        tails = tail_labels(labels,ends)
        np.testing.assert_array_equal(tails,np.lib.stride_tricks.sliding_window_view(labels,5)[ends-5])
        windows.append(pd.DataFrame(dict(subject_id=person,record_id=person,input_start_sec=ends-60,
                                         input_end_sec=ends,target_start_sec=ends-5,target_end_sec=ends,
                                         **{f'target_{i}_label':tails[:,i] for i in range(5)},
                                         valid_target_seconds=np.count_nonzero(tails!=IGNORE,axis=1))))
    all_windows = pd.concat(windows, ignore_index=True)
    body = all_windows.to_csv(index=False,lineterminator='\n').encode()
    if len(all_windows)!=pins['development_manifest_rows'] or audit.digest(body)!=pins['development_manifest_sha256']:
        raise ValueError("重建开发标签与原始归档不一致")
    atomic(MANIFEST,lambda stream:stream.write(body))
    print(f"{len(records)} 名开发受试者，{len(all_windows)} 个窗口，标签字节校验通过（{LABEL_VERSION}）")


def source_identity():
    code_paths = (
        'v3wave/代码/发布复现.py','v3wave/代码/轨道A实验.py','v3wave/代码/B07基座改造实验.py',
        'v3wave/代码/M3首批审计.py','breath5/src/experiment_data.py','breath5/src/nested_split.py',
        'breath5/src/models/e0_five_class.py','breath5/src/models/hierarchical_decoder.py',
        'breath5/src/losses/hierarchical_loss.py','breath5/src/score_runs.py',
        str(SPLIT.relative_to(PROJECT)),'config/外部资产.json',
    )
    external = weights()
    files = [*code_paths,*[str(p.relative_to(PROJECT)) for folder in ('algorithms','third_party/OpenTSLM/src')
                           for p in sorted((PROJECT/'v2.2_ucddb_opentslm'/folder).rglob('*.py'))]]
    return dict(code={p:track.hash_file(PROJECT/p) for p in files}, weights=external, data=require_data(),
                manifest=track.hash_file(MANIFEST), protocol=b07.PROTOCOL)


def training_clock(data,seed):
    return (track.sample_clock(data,'fit',track.STEPS*track.BATCH,seed,True),
            track.sample_clock(data,'select',track.SELECT_SIZE,74121,False))


def make_model(device,seed):
    torch.manual_seed(track.SEEDS[0])
    base = track.E0FiveClass.from_existing(PROJECT,device,compact=True).eval()
    base.boundary.requires_grad_(False)
    initial = {n:p.detach().clone() for n,p in base.named_parameters() if p.requires_grad}
    initial_sha = audit.digest(*(piece for n,p in sorted(initial.items()) for piece in (n.encode(),p.cpu().numpy().tobytes())))
    torch.manual_seed(seed)
    model = b07.configure(base,audit.T03,initial)
    prompt = audit.digest(base.compact_questions.float().cpu().numpy().tobytes())
    return model,initial_sha,prompt


def archive_path():
    value = os.environ.get('BREATHLLM_HISTORICAL_T03')
    return Path(value).expanduser().resolve() if value else LOCAL


def local_db():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    db = sqlite3.connect(LOCAL,timeout=60)
    db.execute('CREATE TABLE IF NOT EXISTS runs (version TEXT, experiment TEXT, fold INT, seed INT, '
               'receipt TEXT, states BLOB, arrays BLOB, PRIMARY KEY(version,experiment,fold,seed))')
    return db


def identity_digest(receipt,states,arrays):
    clean = {k:v for k,v in receipt.items() if k!='content_sha256'}
    return dict(receipt_hash=audit.digest(json.dumps(clean,sort_keys=True).encode()),
                states_sha256=audit.digest(states),arrays_sha256=audit.digest(arrays))


def t03_run(fold,seed):
    path = archive_path()
    if not path.is_file():
        raise FileNotFoundError(f"缺少 T03 归档 {path}。先运行 python breathllm.py t03，或配置经授权的历史库")
    with audit.readonly(path) as db:
        record = db.execute('SELECT receipt, states, arrays FROM runs WHERE version=? AND experiment=? AND fold=? AND seed=?',
                            (audit.VERSION,'T03',fold,seed)).fetchone()
    if record is None:
        raise FileNotFoundError(f"T03 归档缺少内折 {fold}、种子 {seed}。运行相应的 t03 命令后重试")
    entry,states,arrays = record
    receipt = json.loads(entry)
    expected = assets()['historical_t03']['records'][f'{fold}/{seed}'] if path!=LOCAL else receipt['content_sha256']
    if identity_digest(receipt,states,arrays)!=expected:
        raise ValueError(f"归档内容 SHA256 不一致 {fold}/{seed}")
    if receipt['subjects']!=roster()[fold] or receipt['protocol']!=json.loads(json.dumps(b07.PROTOCOL)):
        raise ValueError("归档的划分角色或评分协议不匹配")
    if receipt['spec']!=json.loads(json.dumps(audit.T03)):
        raise ValueError("该归档使用不同的基座配置")
    if path==LOCAL and receipt['source']!=json.loads(json.dumps(source_identity())):
        raise ValueError("当前代码、外部数据或模型与已封存本地归档来源不同")
    return receipt,states,arrays,receipt['source']


def train_t03(folds,seeds,device_name):
    if archive_path()!=LOCAL:
        raise ValueError("运行 t03 前请取消 BREATHLLM_HISTORICAL_T03，训练结果写入本地生成库")
    source = source_identity()
    roles = roster()
    device = torch.device(device_name)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    with closing(local_db()) as db:
        for fold in folds:
            data = track.FoldData(1,roles=roles[fold])
            for seed in seeds:
                exists = db.execute('SELECT 1 FROM runs WHERE version=? AND experiment=? AND fold=? AND seed=?',
                                    (audit.VERSION,'T03',fold,seed)).fetchone()
                if exists:
                    t03_run(fold,seed)
                    print(f"已核验封存内折 {fold} / {seed}",flush=True)
                    continue
                model,init,prompt = make_model(device,seed)
                checkpoint = OUTPUT/'checkpoints'/f't03-{fold}-{seed}.pt'
                run = fit_one(model,data,fold,seed,source,init,prompt,device,checkpoint)
                with db:
                    db.execute('INSERT INTO runs VALUES (?,?,?,?,?,?,?)',(audit.VERSION,'T03',fold,seed,*run))
                checkpoint.unlink(missing_ok=True)
                print(f"已封存内折 {fold} / {seed}",flush=True)


def fit_one(model,data,fold,seed,source,init,prompt,device,checkpoint):
    train,select = training_clock(data,seed)
    batch = lambda role,index: track.batch(data,None,data.roles[role],index,'Base',device)[:2]
    tx,ty = batch('fit',train)
    vx,vy = batch('select',select)
    th = b07.prepare_history(model.base,tx,track.BATCH)
    vh = b07.prepare_history(model.base,vx,track.EVAL_BATCH)
    check_causality(model,tx[:2])
    params = {n:p for n,p in model.named_parameters() if p.requires_grad}
    state_identity = audit.digest(json.dumps(source,sort_keys=True).encode(),
                                  train.tobytes(),select.tobytes(),init.encode(),prompt.encode(),
                                  json.dumps(dict(device=str(device),torch=str(torch.__version__),cuda=torch.version.cuda,
                                                  hardware=torch.cuda.get_device_name(device) if device.type=="cuda" else "cpu"),
                                             sort_keys=True).encode())
    resume = None
    if checkpoint.is_file():
        resume = torch.load(checkpoint,weights_only=True,map_location='cpu')
        if resume["identity"]!=state_identity:
            raise ValueError("T03 恢复点与当前数据、模型或运行环境不一致")
    def seal(state):
        atomic(checkpoint,lambda stream:torch.save(dict(identity=state_identity,**state),stream))
    fit,states = b07.train(model,params,tx,ty,th,None,vx,vy,vh,0.,0.,resume=resume,checkpoint=seal)
    del tx,ty,th,vx,vy,vh
    results,arrays = [],{}
    elapsed = 0.
    for person in data.roles['select']:
        indices = np.arange(len(data.ends[person]))
        x = torch.as_tensor(data.windows(person,indices),device=device)
        blocks,offset = [],0
        for part in np.array_split(indices,max(1,(len(indices)+2047)//2048)):
            interval = slice(offset,offset+len(part))
            blocks.append((interval,b07.prepare_history(model.base,x[interval],track.EVAL_BATCH)))
            offset += len(part)
        rows,p,s = b07.score(model,data,{person:(x,blocks)})
        results.extend(rows)
        arrays.update(p)
        elapsed += s
    saved = io.BytesIO()
    torch.save(states,saved)
    array_data = b07.arrays_to_bytes(dict(**arrays,train_index=train,select_index=select))
    receipt = dict(spec=audit.T03,source=source,protocol=b07.PROTOCOL,subjects=data.roles,
                   fit=fit,results=results,mean=data.mean.tolist(),scale=data.scale.tolist(),
                   scoring_s=elapsed,teacher=None,teacher_sha256=None,teacher_steps=0,
                   trainable=sum(p.numel() for p in params.values()),initial_hash=init,prompt_hash=prompt,
                   compression='npz',disabled_physiology=['b'],optim='AdamW',torch=str(torch.__version__),
                   device=torch.cuda.get_device_name(device) if device.type=='cuda' else str(device),
                   origin='local_rebuild',fold=fold,seed=seed)
    states_data = saved.getvalue()
    receipt['content_sha256'] = identity_digest(receipt,states_data,array_data)
    return json.dumps(receipt,ensure_ascii=False),states_data,array_data


@torch.no_grad()
def check_causality(model,windows):
    lp = model(windows).log_prob
    torch.testing.assert_close(lp.exp().sum(-1),torch.ones_like(lp[...,0]),rtol=0,atol=1e-6)
    for i,stop in enumerate(POSITIONS[:-1],1):
        changed = windows.clone()
        changed[...,stop:] += 11.
        torch.testing.assert_close(model(changed).log_prob[:,:i],lp[:,:i],rtol=0,atol=1e-6)


class Reference:
    def __init__(self, fold, seed, device):
        self.device = torch.device(device)
        self.receipt,self.states,self.arrays,self.source = t03_run(fold,seed)
        if archive_path()!=LOCAL:
            weights()
            require_data()
        self.data = data = track.FoldData(1,roles=roster()[fold])
        np.testing.assert_array_equal(data.mean,self.receipt['mean'])
        np.testing.assert_array_equal(data.scale,self.receipt['scale'])
        self.train,self.select = training_clock(data,seed)
        with np.load(io.BytesIO(self.arrays)) as saved:
            np.testing.assert_array_equal(self.train,saved['train_index'])
            np.testing.assert_array_equal(self.select,saved['select_index'])
        self.model,initial,prompt = make_model(self.device,seed)
        if (initial,prompt)!=(self.receipt['initial_hash'],self.receipt['prompt_hash']):
            raise ValueError('T03 初始化或提示词哈希不一致')
        self.params = {n:p for n,p in self.model.named_parameters() if p.requires_grad}
        states = torch.load(io.BytesIO(self.states),weights_only=True,map_location='cpu')
        for name,p in self.params.items():
            torch.testing.assert_close(p.cpu(),states[0][name],atol=0,rtol=0)
        self.best = states[self.receipt['fit']['best_step']]
        self.restore(self.best)
        self.model.eval()

    def weights(self):
        return {n:p.detach().to('cpu',copy=True) for n,p in self.params.items()}

    @torch.no_grad()
    def restore(self,state):
        if state.keys()!=self.params.keys():
            raise ValueError('T03 冻结状态的张量键不一致')
        for name,value in self.params.items():
            value.copy_(state[name])

    def batch(self,role,indices):
        return track.batch(self.data,None,self.data.roles[role],indices,'Base',self.device)[:2]

    @torch.no_grad()
    def replay(self):
        person = self.data.roles['select'][0]
        idx = np.array_split(np.arange(len(self.data.ends[person])),
                             (len(self.data.ends[person])+2047)//2048)[0][:track.EVAL_BATCH]
        x,_ = self.batch('select',np.column_stack((idx*0,idx)))
        p = b07.predict(self.model,x,b07.prepare_history(self.model.base,x,track.EVAL_BATCH))
        with np.load(io.BytesIO(self.arrays)) as data:
            original = data[person][idx]
        np.testing.assert_allclose(p,original,rtol=0,atol=audit.ATOL)
        return dict(windows=len(idx),max_gap=float(np.abs(p-original).max()))


def reference():
    torch.set_float32_matmul_precision('high')
    ref = Reference(1,4121,DEVICE)
    identity = dict(fold=1,seed=4121,t03=audit.VERSION,source=ref.source,
                    states=audit.digest(ref.states),arrays=audit.digest(ref.arrays))
    return ref,identity,ref.replay()


@torch.no_grad()
def native_encoding(ref,x,positions):
    states = []
    handle = ref.model.base.decoder.register_forward_pre_hook(lambda _,args:states.append(torch.cat(args,-1)))
    try:
        result = ref.model(x,ref.model.base.fold_local_features(x))
    finally:
        handle.remove()
    assert len(states)==1
    out = tuple(v[:,-positions:].contiguous().cpu() for v in (states[0],*result.conditional_logits))
    assert all(v.isfinite().all() for v in out)
    return out


def encoding_cache(ref,identity,role,index,positions,name,limit):
    code = track.hash_file(Path(__file__))
    key = audit.digest(json.dumps(identity,sort_keys=True).encode(),code.encode(),index.tobytes(),
                       str(positions).encode(),str(ref.device).encode(),str(torch.__version__).encode())[:24]
    directory = OUTPUT/'cache/r1'/key
    directory.mkdir(parents=True,exist_ok=True)
    size = track.BATCH if role=='fit' else track.EVAL_BATCH
    parts = np.array_split(np.arange(len(index)),max(1,(len(index)+2047)//2048)) if role=='select' else [np.arange(len(index))]
    blocks = []
    for part in parts:
        for rows in np.split(part,np.arange(size,len(part),size)):
            limit()
            path = directory/f'{name}-{rows[0]}.pt'
            if not path.exists():
                x,_ = ref.batch(role,index[rows])
                encoded = native_encoding(ref,x,positions)
                atomic(path,lambda stream:torch.save(dict(index=torch.from_numpy(index[rows]),encoded=encoded),stream))
            saved = torch.load(path,weights_only=True,map_location='cpu')
            np.testing.assert_array_equal(saved['index'].numpy(),index[rows])
            blocks.append(saved['encoded'])
    return tuple(map(torch.cat,zip(*blocks,strict=True))),key


def real_smoke(device_name):
    require_data()
    weights()
    torch.set_float32_matmul_precision('high')
    device = torch.device(device_name)
    data = track.FoldData(1,roles=roster()[1])
    index = track.sample_clock(data,'fit',2,4121,True)
    x,y,_ = track.batch(data,None,data.roles['fit'],index,'Base',device)
    model,initial,prompt = make_model(device,4121)
    model.eval()
    check_causality(model,x)
    if archive_path().exists():
        receipt,_,_,_ = t03_run(1,4121)
        assert (initial,prompt)==(receipt['initial_hash'],receipt['prompt_hash'])
    with torch.no_grad():
        result = model(x).log_prob.exp()
    assert result.shape == (2,5,5) and y.shape==(2,5)
    print('两个真实开发窗的 T03 初始化、五类概率和末五位置因果检查通过')
