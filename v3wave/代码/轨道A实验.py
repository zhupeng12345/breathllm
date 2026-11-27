"""两种冻结 OpenTSLM 基座与三个单模块的轨道 A 验证。"""

import argparse
import io
import json
import sqlite3
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from numba import njit
from torch import nn
from torch.nn import functional as F

from breathllm_m3 import ConditionalEvidence, DynamicsPrompt, track_a_evidence


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from breathllm_paths import MANIFEST, OUTPUT, PROJECT

BREATH = PROJECT / "breath5"
sys.path.insert(0, str(BREATH))

from src.experiment_data import FoldData, PSG
from src.losses.hierarchical_loss import classification_loss
from src.models.e0_five_class import E0FiveClass, ForwardResult
from src.models.hierarchical_decoder import five_class_log_probs
from src.nested_split import SPLIT, fold_folder, roles_for
from hashlib import file_digest
from src.score_runs import totals


SEEDS = (4121, 4122, 4123)
ARMS = ("Base", "WD-AMT", "MC-DPT", "DT-EMM")
STEPS, CHECK, PATIENCE, BATCH = 512, 128, 2, 32
SELECT_SIZE, EVAL_BATCH, LR = 8192, 512, 2e-4


def hash_file(path):
    with Path(path).open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def route(split, outer, inners):
    partitions = [roles_for(split, outer, i) for i in inners]
    reserved = partitions[0]["reserved"]
    members = set(partitions[0]["fit"] + partitions[0]["select"])
    all_select = []
    if len(reserved) != 5 or len(members) != 20:
        raise ValueError("Expected 20 development and five reserved participants")
    for roles in partitions:
        if roles["reserved"] != reserved or roles["score"]:
            raise ValueError("An inner split uses outer participants")
        fit, select = set(roles["fit"]), set(roles["select"])
        if fit & select or (fit | select) != members or members.intersection(reserved):
            raise ValueError("Partition membership differs or is not disjoint")
        all_select.extend(roles["select"])
    if sorted(all_select) != sorted(members):
        raise ValueError("Development subjects must validate exactly once")
    return dict(zip(inners, partitions))


@njit
def memory_series(blocks, mechanisms):
    """One write per second. Read normalized slots plus support and direction."""
    result = np.zeros((len(blocks), 67), dtype=np.float32)
    central = np.zeros(32, dtype=np.float32)
    obstructive = central.copy()
    c = o = co = oc = 0.
    for i in range(len(blocks)):
        wc, wo = mechanisms[i]
        pause = 1 - np.exp(-wc - wo)
        rho = np.exp(-1 / 20.) * (1 - .25 * (1 - pause))
        co = rho * co + c * wo
        oc = rho * oc + o * wc
        c, o = rho * c + wc, rho * o + wo
        for j in range(32):
            central[j] = rho * central[j] + wc * blocks[i, j]
            obstructive[j] = rho * obstructive[j] + wo * blocks[i, j]
        result[i, :32] = central / max(c, 1e-5)
        result[i, 32:64] = obstructive / max(o, 1e-5)
        result[i, 64] = np.log1p(c)
        result[i, 65] = np.log1p(o)
        result[i, 66] = (co - oc) / (co + oc + 1e-5)
    return result


@torch.no_grad()
def evidence_by_second(signal, device):
    windows = np.lib.stride_tricks.sliding_window_view(signal, 55 * 8, axis=1)[:, ::8]
    dynamics, relations = [], []
    for start in range(0, windows.shape[1], 1024):
        block = windows[:, start:start + 1024].transpose(1, 0, 2).copy()
        a, b = track_a_evidence(torch.as_tensor(block, device=device))
        dynamics.append(a.cpu().numpy())
        relations.append(b.cpu().numpy())
    a, b = np.concatenate(dynamics), np.concatenate(relations)
    blocks = np.ascontiguousarray(windows[..., -8:].transpose(1, 0, 2)).reshape(-1, 32)
    c = memory_series(blocks.clip(-8, 8), np.ascontiguousarray(b[:, 0, 3:5]))
    return {"WD-AMT": a, "MC-DPT": b.reshape(len(b), -1), "DT-EMM": c}


class TrackModel(nn.Module):
    def __init__(self, base, kind, arm):
        super().__init__()
        self.base, self.kind, self.arm = base, kind, arm
        width = base.backbone.questions.size(-1)
        self.head = nn.Linear(width, 5) if kind == "B06" else None
        if arm == "WD-AMT":
            self.addition = DynamicsPrompt(width)
        elif arm in ("MC-DPT", "DT-EMM"):
            self.addition = ConditionalEvidence(27 if arm == "MC-DPT" else 67, width)

    def forward(self, x, evidence):
        hook = None
        if self.arm == "WD-AMT":
            def inject(_layer, args, kwargs):
                prompt = kwargs["inputs_embeds"]
                tail = prompt[:, -15:].reshape(len(x), 5, 3, -1)
                local = tail[:, :, 0]
                tail = torch.cat(((local + self.addition(local, evidence, x)).unsqueeze(2), tail[:, :, 1:]), 2)
                return args, dict(kwargs, inputs_embeds=torch.cat((prompt[:, :-15], tail.flatten(1, 2)), 1))

            decoder = self.base.backbone.llm.get_base_model().model
            hook = decoder.register_forward_pre_hook(inject, with_kwargs=True)
        try:
            event, subtype, physiology = self.base.encode_states(x)
        finally:
            if hook is not None:
                hook.remove()
        delta = self.addition(subtype, evidence) if self.arm in ("MC-DPT", "DT-EMM") else 0.
        if self.head is not None:
            logits = self.head(subtype).float()
            apnea = logits[..., 1:4]
            adjusted = (apnea + delta).log_softmax(-1) + apnea.logsumexp(-1, keepdim=True)
            return torch.cat((logits[..., :1], adjusted, logits[..., 4:]), -1)
        primary = self.base.decoder(event, subtype)
        expert = self.base.physiology(physiology, physiology)
        a, b, q = tuple((x1 + x2) * .5 for x1, x2 in zip(primary, expert))
        q = q + delta
        return ForwardResult(five_class_log_probs(a, b, q), (a, b, q), None)


def sample_clock(data, role, size, seed, replace):
    people = data.roles[role]
    clock = np.concatenate([np.column_stack((np.full(len(take), i), take))
                            for i, person in enumerate(people)
                            for take in [np.flatnonzero(data.labels[person][:, -1] >= 0)]])
    index = np.random.default_rng(seed).choice(len(clock), size, replace=replace)
    return clock[index]


def batch(data, evidence, people, clocks, arm, device):
    count = len(clocks)
    x = np.empty((count, 4, 480), dtype=np.float32)
    y = np.empty((count, 5), dtype=np.int64)
    features = None
    if arm != "Base":
        shape = evidence[people[0]][arm].shape[1:]
        features = np.empty((count, 5, *shape), dtype=np.float32)
    for i in np.unique(clocks[:, 0]):
        mask, person = clocks[:, 0] == i, people[i]
        index = clocks[mask, 1]
        x[mask], y[mask] = data.windows(person, index), data.labels[person][index]
        if features is not None:
            features[mask] = evidence[person][arm][index[:, None] + np.arange(1, 6)]
    return tuple(torch.as_tensor(value, device=device) if value is not None else None
                 for value in (x, y, features))


def log_probs(output):
    return output.log_softmax(-1) if isinstance(output, torch.Tensor) else output.log_prob


def confusion(y, prediction):
    valid = y >= 0
    return np.bincount(y[valid].astype(int) * 5 + prediction[valid], minlength=25).reshape(5, 5)


def macro_f1(matrix):
    return float((2 * matrix.diagonal() / (matrix.sum(0) + matrix.sum(1)).clip(1)).mean())


@torch.no_grad()
def probabilities(model, x, features):
    return torch.cat([
        log_probs(model(x[start:start + EVAL_BATCH],
                        None if features is None else features[start:start + EVAL_BATCH]))[:, -1].exp().cpu()
        for start in range(0, len(x), EVAL_BATCH)
    ]).numpy()


def fit(model, teacher, data, evidence, seed, selected, device):
    train = sample_clock(data, "fit", STEPS * BATCH, seed, True)
    x, y, e = batch(data, evidence, data.roles["select"], selected, model.arm, device)
    validation = y[:, -1].cpu().numpy()
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(params.values(), lr=LR)
    elapsed, best, curve = 0., (-1., 0), []
    for check in range(CHECK, STEPS + 1, CHECK):
        start = perf_counter()
        running = torch.zeros((), device=device)
        for step in range(check - CHECK, check):
            clocks = train[step * BATCH:(step + 1) * BATCH]
            signal, target, extra = batch(data, evidence, data.roles["fit"], clocks, model.arm, device)
            output = model(signal, extra)
            if teacher is not None:
                with torch.no_grad():
                    guidance = teacher(signal)
                loss = classification_loss(output, target, teacher_logits=guidance)
            else:
                loss = F.nll_loss(log_probs(output).flatten(0, 1), target.flatten(), ignore_index=-1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params.values(), 1.)
            optimizer.step()
            running += loss.detach()
        train_loss = (running / CHECK).item()
        if not np.isfinite(train_loss):
            raise FloatingPointError(f"{model.kind} {model.arm} {seed}: nonfinite loss at {check}")
        elapsed += perf_counter() - start
        score = macro_f1(confusion(validation, probabilities(model, x, e).argmax(-1)))
        curve.append(dict(step=check, loss=train_loss, macro_f1=score, training_s=elapsed))
        if score > best[0]:
            best = score, check
            saved = {name: p.detach().cpu().clone() for name, p in params.items()}
        print(f"{model.kind} {model.arm} {seed} step={check} inner_macro={score:.4f} train_s={elapsed:.1f}", flush=True)
        if check - best[1] >= CHECK * PATIENCE:
            break
    with torch.no_grad():
        for name, value in saved.items():
            params[name].copy_(value)
    return dict(best_step=best[1], curve=curve, trainable=sum(p.numel() for p in params.values())), saved, train


def score_nights(model, data, evidence, device):
    rows, arrays = [], {}
    people = data.roles["select"]
    for i, person in enumerate(people):
        start = perf_counter()
        indexes = np.arange(len(data.ends[person]))
        clocks = np.column_stack((np.full(len(indexes), i), indexes))
        parts = []
        for sub in np.array_split(clocks, max(1, (len(clocks) + 2047) // 2048)):
            x, _, e = batch(data, evidence, people, sub, model.arm, device)
            parts.append(probabilities(model, x, e))
        p = np.concatenate(parts)
        if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1., atol=2e-5):
            raise FloatingPointError(f"Invalid prediction for {person}")
        y = data.labels[person][:, -1]
        counts, matched = totals(y, p, .5, details=True, detect_first=True, gap=5, min_run=5)
        support = [int((matched[:, 2] == c).sum()) for c in range(1, 5)]
        hits = [int(((matched[:, 2] == c) & (matched[:, 3] == c)).sum()) for c in range(1, 5)]
        rows.append(dict(person=person, cm=confusion(y, p.argmax(-1)).tolist(), counts=counts,
                         event_support=support, event_hits=hits, inference_s=perf_counter() - start))
        arrays[f"{person}_second"] = data.ends[person] - 1
        arrays[person] = p
    return rows, arrays


def check_causality(model, data, evidence, device):
    person = data.roles["fit"][0]
    x, _, e = batch(data, evidence, [person], np.array([[0, 0]]), model.arm, device)
    with torch.no_grad():
        first = log_probs(model(x, e))
        changed = x.clone()
        changed[..., -8:] += 11
        extra = None if e is None else e.clone()
        if extra is not None:
            extra[:, -1] += 3
            model.addition.output.weight.fill_(.01)
            first = log_probs(model(x, e))
        second = log_probs(model(changed, extra))
        torch.testing.assert_close(first[:, :-1], second[:, :-1], atol=1e-6, rtol=0.)
        if extra is not None:
            model.addition.output.weight.zero_()


if __name__ == "__main__":
    raise SystemExit("统一入口为 python breathllm.py t03 --help")
