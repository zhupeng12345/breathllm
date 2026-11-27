"""不依赖真实数据和预训练权重的交接验收。"""

import importlib
import io
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

import 发布复现 as repro
from src.data.sliding_window_60s import tail_labels, window_ends
from src.labels.second_level_labels import rasterize
from src.models.e0_five_class import E0FiveClass
from src.models.hierarchical_decoder import five_class_log_probs


def synthetic_backbone():
    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Conv1d(1, 8, 4, stride=4)
            self.pos_embed = torch.zeros(1, 60, 8)
            self.input_norm = nn.Identity()
            self.encoder = SimpleNamespace(layers=[nn.Identity()])

    class TinyCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(6, 8)
            self.model = nn.Module()
            self.model.forward = lambda inputs_embeds, **_: SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(1))

        def get_base_model(self):
            return self

        def get_output_embeddings(self):
            return self.embedding

    base = nn.Module()
    base.encoder = TinyEncoder()
    base.llm = TinyCausalLM()
    base.questions = torch.randn(1, 8, 8)
    base.first_answer = 3
    base.question_answers = torch.tensor([[0, 1], [2, 3]])
    base.fixed_prefix = torch.randn(1, 2, 8)
    base.projector = nn.Linear(8, 8)
    base.channel_embedding = nn.Parameter(torch.zeros(4, 8))
    base.amplitude_projection = nn.Parameter(torch.ones(8))
    return base


def train_resume():
    class Readout(nn.Linear):
        def forward(self, x, history=None):
            logits = super().forward(x)
            a,b,q = logits[...,0],logits[...,1],logits[...,2:]
            return repro.track.ForwardResult(five_class_log_probs(a,b,q),(a,b,q),None)

    torch.manual_seed(4121)
    data = torch.randn(8,5,4)
    labels = torch.arange(5).expand(len(data),-1)
    example = Readout(4,5).state_dict()
    paths = {}
    def seal(snapshot):
        buffer = io.BytesIO()
        torch.save(snapshot,buffer)
        paths[snapshot['step']] = buffer.getvalue()
    def run_fit(resume=None):
        model = Readout(4,5)
        model.load_state_dict(example)
        parameters = dict(model.named_parameters())
        with patch.multiple(repro.track,STEPS=4,CHECK=2,BATCH=2,PATIENCE=10):
            return repro.b07.train(model,parameters,data,labels,(),None,data,labels,(),0.,0.,
                                   checkpoint=seal,resume=resume)
    _,expected = run_fit()
    final = torch.load(io.BytesIO(paths[4]),weights_only=True)
    state = torch.load(io.BytesIO(paths[2]),weights_only=True)
    _,restored = run_fit(resume=state)
    replay = torch.load(io.BytesIO(paths[4]),weights_only=True)
    for name in expected[4]:
        torch.testing.assert_close(expected[4][name],restored[4][name],atol=0,rtol=0)
    for parameter,saved in final['optimizer']['state'].items():
        for name,value in saved.items():
            torch.testing.assert_close(replay['optimizer']['state'][parameter][name],value,atol=0,rtol=0)
    torch.testing.assert_close(final['rng'],replay['rng'],atol=0,rtol=0)



def run():
    torch.set_num_threads(2)
    torch.manual_seed(4121)
    assert set(repro.roster()) == {1, 2, 3}
    assert [len(v['select']) for v in repro.roster().values()] == [7, 7, 6]
    stages = np.array([1, 1, 1, 0])
    events = pd.DataFrame({'start_sec':[61.0,75.0], 'end_sec':[70.0,84.0],
                           'original_label':['APNEA-O','HYP-C'], 'periodic_flag':['','']})
    y = rasterize(120,stages,events).labels
    tails = tail_labels(y,window_ends(120))
    assert y[61]==1 and y[75]==4 and y[90]==-1
    assert tails.shape==(61,5) and np.array_equal(tails[0],y[55:60])
    inputs = torch.randn(2,4,480)
    model = E0FiveClass(synthetic_backbone(),compact=True).eval()
    repro.check_causality(model,inputs)
    with torch.no_grad():
        h, subtype, phys = model.encode_states(inputs)
        assert h.shape==subtype.shape==(2,5,8) and phys.shape==(2,5,10)
    logits = torch.randn(8,5)
    p = five_class_log_probs(logits[:,0],logits[:,1],logits[:,2:]).exp()
    torch.testing.assert_close(p.sum(1),torch.ones(8),atol=1e-6,rtol=0)
    truth = np.zeros(32,dtype=int)
    truth[2:9] = 4
    truth[20:26] = 2
    p = np.eye(5)[np.where(truth==2, 1, truth)]
    p[6:8] = np.eye(5)[0]
    counts,pos,match = repro.b07.event_score(truth,p)
    repro.audit.check_matches(truth,pos,match)
    assert (counts['predicted'],counts['detected'],counts['typed'])==(2,2,1)
    fixed = repro.audit.fixed_labels(match,pos,np.eye(5)[truth])
    np.testing.assert_array_equal(fixed,[4,2])
    baseline = importlib.import_module('60s五分类baseline')
    assert len(baseline.METHODS)==20 and len(set(baseline.METHODS))==20
    for method in baseline.METHODS:
        if method in baseline.TABULAR:
            signals = np.random.default_rng(4121).normal(size=(2,4,480)).astype(np.float32)
            signals[:,3] += 96
            dummy = SimpleNamespace(scale=np.ones(4),mean=np.zeros(4))
            z = baseline.tabular_features(method,signals,dummy)
            assert len(z)==2 and np.isfinite(z).all()
        else:
            with torch.no_grad():
                z = baseline.NeuralBaseline(method).eval()(inputs,None)
            assert z.shape[0]==2 and z.shape[-1]==5 and z.isfinite().all()
    r1 = importlib.import_module('BreathLLM-R1')
    h = torch.randn(64,5,54)
    y = torch.tensor([0,1,4,2,3]).expand(64,-1)
    group = torch.arange(64)[:,None].expand_as(y)
    a,b,q = torch.randn(64,5),torch.randn(64,5),torch.randn(64,5,3)
    models = {n:r1.Residual(54) for n in ('R03','R04','R05')}
    models.update(R01=r1.Calibration(False),R02=r1.Calibration(True))
    assert len(r1.mathematical_audit(models,h,(a,b,q),y,group,repro.OPT))>=5
    r3 = importlib.import_module('BreathLLM-R3-PAEL')
    assert len(r3.math_tests())>=5
    train_resume()
    print('通过：折划分、RESP 标签、末五位置因果性、五类概率、固定匹配、20 种基线接口、R1 与 PAEL 数学检查、T03 AdamW 恢复')
