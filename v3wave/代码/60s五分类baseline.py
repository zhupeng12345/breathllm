"""Historical UCDDB baselines adapted to the v3wave Track A protocol."""

import argparse
import io
import json
import sqlite3
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F
from xgboost import XGBClassifier

import 轨道A实验 as track
from src.classical import make_forest
from src.models.cnn_lstm_teacher import CnnLstmTeacher
from src.models.resnet1d import ResNet1D
from src.models.tcn import TCN
from src.models.yu_lstm_cnn import YuLSTMCNN
from src.models.cheng_mil import ChengMIL


LITERATURE = track.PROJECT / "breathnas/v3/literature_benchmark"
LEGACY = track.PROJECT / "v1.0/experiments"
sys.path.extend((str(LITERATURE), str(LEGACY)))
from methods import evoapnea, mahmud_cnn, respnet, spitrannet, ma_svm, saha_knn, yu_rf
from algorithms import cnn1d, inceptiontime, aix_tsdnet_3c


FRAME_MODELS = ("CNN-LSTM", "CNN-LSTM+persistence", "TCN", "ResNet1D", "Yu LSTM-CNN", "Cheng MIL")
PAPER_NETWORKS = {"Mahmud EEG-CNN": mahmud_cnn, "RespNet": respnet,
                  "SpiTranNet": spitrannet, "EvoApneaFormer": evoapnea}
WINDOW_MODELS = (*PAPER_NETWORKS, "1D-CNN", "InceptionTime", "AIX-TSDNet", "AIX-MultiChannel")
TABULAR = ("Random Forest", "XGBoost", "SVM", "Ma RBF-SVM", "Saha KNN", "Yu cascade RF")
METHODS = (*FRAME_MODELS, *WINDOW_MODELS, *TABULAR)
MODE = dict(channels=["airflow", "thoracic", "abdominal", "spo2"], hz=8)
YU_COARSE_WIDTH = 10


class AIX(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.encoders = nn.ModuleList(aix_tsdnet_3c.TSDNet3C(i) for i in channels)
        for encoder in self.encoders:
            encoder.classifier = nn.Identity()
        self.head = nn.Linear(8 * len(channels), 5)

    def forward(self, x):
        x = F.interpolate(x, size=256, mode="linear", align_corners=False)
        return self.head(torch.cat([encode(x).flatten(1) for encode in self.encoders], dim=1))


class NeuralBaseline(nn.Module):
    arm = "Base"

    def __init__(self, method):
        super().__init__()
        self.kind = method
        if method in FRAME_MODELS:
            factories = {"CNN-LSTM": CnnLstmTeacher, "CNN-LSTM+persistence": CnnLstmTeacher,
                         "TCN": TCN, "ResNet1D": ResNet1D, "Yu LSTM-CNN": YuLSTMCNN, "Cheng MIL": ChengMIL}
            self.network = factories[method](4)
            if method == "CNN-LSTM+persistence":
                self.persistence = nn.Linear(4, 5, bias=False)
                nn.init.zeros_(self.persistence.weight)
        elif method in PAPER_NETWORKS:
            self.network = PAPER_NETWORKS[method].Model(MODE)
            if method == "EvoApneaFormer":
                self.network.subtype = nn.Linear(128, 3)
            elif method == "SpiTranNet":
                self.network.head = nn.Linear(64, 5)
            else:
                old = self.network.head[-1]
                self.network.head[-1] = nn.Linear(old.in_features, 5)
        elif method == "1D-CNN":
            self.network = cnn1d.ConvNet()
            self.network.fc = nn.Linear(96, 5)
        elif method == "InceptionTime":
            self.network = inceptiontime.build_inceptiontime(4)
            self.network[-1] = nn.Linear(128, 5)
        else:
            self.network = AIX((0,) if method == "AIX-TSDNet" else (0, 1, 3))

    def forward(self, x, _):
        if self.kind in FRAME_MODELS:
            logits = self.network(x)
            if self.kind == "CNN-LSTM+persistence":
                flow = x[:, 0].reshape(-1, 60, 8)
                amplitude = flow.amax(-1) - flow.amin(-1)
                reference = amplitude[:, :30].median(-1).values.clamp_min(1e-4)
                relative = (amplitude / reference[:, None]).clamp(0, 4)
                history = relative.unfold(-1, 20, 1)[:, -5:]
                evidence = torch.stack((history[..., -5:].mean(-1), history.mean(-1),
                                        (history < .7).float().mean(-1), (history < .3).float().mean(-1)), -1)
                logits = logits + self.persistence(evidence)
            return logits
        if not torch.is_grad_enabled():
            return self.network(x)[:, None]
        return torch.stack([self.network(x[..., :end]) for end in range(448, 481, 8)], dim=1)


def engineered(x):
    blocks = []
    for width in (5, 10, 30, x.shape[-1] // 8):
        wave = x[..., -8 * width:]
        blocks.extend((wave.mean(2), wave.std(2), np.ptp(wave, axis=2),
                       np.abs(np.diff(wave, axis=2)).mean(2)))
        if width in (10, 30):
            centred = wave[:, :3] - wave[:, :3].mean(2, keepdims=True)
            power = np.abs(np.fft.rfft(centred * np.hanning(8 * width), axis=2)) ** 2
            hz = np.fft.rfftfreq(8 * width, 1 / 8)
            total = power[:, :, 1:].sum(2).clip(1e-12)
            for lo, hi in ((.1, .25), (.25, .5), (.5, 1.)):
                blocks.append(power[:, :, (hz >= lo) & (hz < hi)].sum(2) / total)
            norms = np.linalg.norm(centred, axis=-1)
            for a, b in ((0, 1), (0, 2), (1, 2)):
                blocks.append((centred[:, a] * centred[:, b]).sum(1, keepdims=True)
                              / (norms[:, a] * norms[:, b]).clip(1e-8)[:, None])
    flow = np.ptp(x[:, 0].reshape(len(x), -1, 8), axis=-1)
    oxygen = x[:, 3].reshape(len(x), -1, 8).mean(2)
    ratio = flow[:, -20:] / flow[:, :30].max(1, keepdims=True).clip(1e-5)
    blocks.extend(((ratio < .7).mean(1, keepdims=True), (ratio < .3).mean(1, keepdims=True),
                   (oxygen[:, :30].max(1) - oxygen[:, -10:].mean(1))[:, None],
                   (oxygen[:, -1] - oxygen[:, -10])[:, None]))
    return np.column_stack(blocks).astype(np.float32)


def tabular_features(method, x, data):
    if method == "Ma RBF-SVM":
        # The historical slope feature uses five equal temporal bins.
        seconds = x.shape[-1] // 8
        padded = np.pad(x, ((0, 0), (0, 0), ((-seconds % 5) * 8, 0)), mode="edge")
        return ma_svm.features(padded, 8).astype(np.float32)
    if method == "Saha KNN":
        return saha_knn.features(x, MODE)
    if method == "Yu cascade RF":
        x = x * data.scale[None, :, None] + data.mean[None, :, None]
        return np.column_stack((yu_rf.features(x, MODE, True), yu_rf.features(x[..., -80:], MODE, False)))
    return engineered(x)


class TabularBaseline:
    def __init__(self, method, seed, device):
        self.method, self.seed, self.device = method, seed, device
        self.scaler = MinMaxScaler() if method == "Ma RBF-SVM" else StandardScaler()

    def fit(self, x, y, validation, targets):
        start = perf_counter()
        method = self.method
        forest = method in ("Random Forest", "Yu cascade RF")
        growing = forest or method == "XGBoost"
        if method == "Saha KNN":
            self.train = torch.as_tensor(x, device=self.device)
            self.labels = torch.as_tensor(y, device=self.device)
        elif method in ("Ma RBF-SVM", "SVM"):
            a, b = (1000, .02) if method == "Ma RBF-SVM" else (2, "scale")
            self.svm = SVC(C=a, gamma=b, cache_size=768, decision_function_shape="ovo")
            self.svm.fit(self.scaler.fit_transform(x), y)
            self.svm_cache()
        elif forest:
            self.coarse = make_forest(self.seed, 5, 1)
            self.coarse.set_params(warm_start=True)
            if method == "Yu cascade RF":
                self.fine = make_forest(self.seed + 1, 5, 1)
                self.fine.set_params(warm_start=True)

        budgets = {"Random Forest": (30, 60, 90, 120), "XGBoost": (50, 100, 150, 200),
                   "Yu cascade RF": (32, 64, 96, 128), "Saha KNN": (5, 17, 41)}
        best, best_at, best_model, curve = -1., 0, None, []
        continuation = None
        for i, budget in enumerate(budgets.get(method, (1,))):
            if forest:
                cut = YU_COARSE_WIDTH if method == "Yu cascade RF" else x.shape[1]
                coarse_y = (y > 0).astype(int) if method == "Yu cascade RF" else y
                coarse_trees = budget // 2 if method == "Yu cascade RF" else budget
                self.coarse.set_params(n_estimators=coarse_trees).fit(x[:, :cut], coarse_y)
                if method == "Yu cascade RF":
                    self.fine.set_params(n_estimators=budget).fit(x[:, cut:], y)
            elif method == "XGBoost":
                self.booster = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=.06,
                                             subsample=.8, colsample_bytree=.8, n_jobs=5, tree_method="hist",
                                             eval_metric="mlogloss", random_state=self.seed)
                self.booster.fit(x, y, xgb_model=continuation)
                continuation = self.booster.get_booster()
            elif method == "Saha KNN":
                self.k = budget
            elapsed = perf_counter() - start
            p = self.predict(validation)
            f1 = track.macro_f1(track.confusion(targets, p.argmax(1)))
            curve.append(dict(budget=budget, macro_f1=f1, seconds=elapsed))
            if f1 > best:
                best, best_at = f1, i
                best_model = self.export()
            if growing and i - best_at >= track.PATIENCE:
                break
        self.restore(best_model)
        return dict(best=curve[best_at]["budget"], curve=curve), best_model

    def export(self):
        names = {"Saha KNN": ("k", "train", "labels"), "Ma RBF-SVM": ("svm", "scaler"),
                 "SVM": ("svm", "scaler"), "XGBoost": ("booster",),
                 "Random Forest": ("coarse",), "Yu cascade RF": ("coarse", "fine")}
        state = {name: getattr(self, name) for name in names[self.method]}
        if self.method == "Saha KNN":
            state.update(train=self.train.cpu().numpy(), labels=self.labels.cpu().numpy())
        output = io.BytesIO()
        joblib.dump(state, output, compress=3)
        return output.getvalue()

    def restore(self, state):
        self.__dict__.update(joblib.load(io.BytesIO(state)))
        if self.method in ("SVM", "Ma RBF-SVM"):
            self.svm_cache()
        elif self.method == "Saha KNN":
            self.train = torch.as_tensor(self.train, device=self.device)
            self.labels = torch.as_tensor(self.labels, device=self.device)

    def svm_cache(self):
        model = self.svm
        self.support = torch.tensor(model.support_vectors_, dtype=torch.float32, device=self.device)
        classes = np.repeat(np.arange(5), model.n_support_)
        coefficients = np.zeros((len(classes), 10), dtype=np.float32)
        pairs = []
        for i in range(5):
            for j in range(i + 1, 5):
                coefficients[classes == i, len(pairs)] = model.dual_coef_[j - 1, classes == i]
                coefficients[classes == j, len(pairs)] = model.dual_coef_[i, classes == j]
                pairs.append(np.eye(5)[i] - np.eye(5)[j])
        self.coeff = torch.as_tensor(coefficients, device=self.device)
        self.pairs = np.array(pairs)

    @torch.no_grad()
    def predict(self, features):
        method = self.method
        if method == "Random Forest":
            return self.coarse.predict_proba(features)
        if method == "XGBoost":
            return self.booster.predict_proba(features)
        if method == "Yu cascade RF":
            coarse = self.coarse.predict_proba(features[:, :YU_COARSE_WIDTH])[:, 1]
            fine = self.fine.predict_proba(features[:, YU_COARSE_WIDTH:])
            abnormal = fine[:, 1:].sum(1)
            odds = np.sqrt((coarse.clip(1e-6) / (1-coarse).clip(1e-6))
                           * (abnormal.clip(1e-6) / (1-abnormal).clip(1e-6)))
            mass = odds / (1 + odds)
            conditional = np.divide(fine[:, 1:], abnormal[:, None], out=np.full_like(fine[:, 1:], .25),
                                    where=abnormal[:, None] > 0)
            return np.column_stack((1 - mass, mass[:, None] * conditional))
        if method == "Saha KNN":
            result = []
            for x in np.array_split(features, max(1, (len(features) + 255) // 256)):
                values, ids = (torch.as_tensor(x, device=self.device) @ self.train.T).topk(self.k)
                weights = (1 - values.clamp(-1, 1) + .04).reciprocal()
                votes = (F.one_hot(self.labels[ids], 5) * weights[..., None]).sum(1) + .01
                result.append((votes / votes.sum(1, keepdim=True)).cpu().numpy())
            return np.concatenate(result)
        features = self.scaler.transform(features).astype(np.float32)
        result = []
        support_norm = self.support.square().sum(1)
        for part in np.array_split(features, max(1, (len(features) + 255) // 256)):
            x = torch.as_tensor(part, device=self.device)
            distance = (x.square().sum(1, keepdim=True) + support_norm - 2 * (x @ self.support.T)).clamp_min_(0)
            margins = torch.exp(-self.svm._gamma * distance) @ self.coeff
            result.append(margins.cpu().numpy() + self.svm.intercept_)
        margins = np.concatenate(result)
        return softmax((margins @ self.pairs) / 4, axis=1)


def sampled_features(data, role, clocks, method, train):
    x, y, _ = track.batch(data, None, data.roles[role], clocks, "Base", "cpu")
    x, y = x.numpy(), y.numpy()
    if not train:
        return tabular_features(method, x, data), y[:, -1]
    blocks, targets = [], []
    for i, end in enumerate(range(448, 481, 8)):
        valid = y[:, i] >= 0
        blocks.append(tabular_features(method, x[valid, :, :end], data))
        targets.append(y[valid, i])
    return np.concatenate(blocks), np.concatenate(targets)


def tabular_nights(model, data, method):
    rows, arrays = [], {}
    for person in data.roles["select"]:
        start = perf_counter()
        idx = np.arange(len(data.ends[person]))
        parts = [model.predict(tabular_features(method, data.windows(person, take), data))
                 for take in np.array_split(idx, max(1, (len(idx) + 1023) // 1024))]
        p = np.concatenate(parts).astype(np.float32)
        np.testing.assert_allclose(p.sum(1), 1., atol=2e-5)
        y = data.labels[person][:, -1]
        counts, matched = track.totals(y, p, .5, details=True, detect_first=True, gap=5, min_run=5)
        support = [int((matched[:, 2] == c).sum()) for c in range(1, 5)]
        hits = [int(((matched[:, 2] == c) & (matched[:, 3] == c)).sum()) for c in range(1, 5)]
        rows.append(dict(person=person, cm=track.confusion(y, p.argmax(1)).tolist(), counts=counts,
                         event_support=support, event_hits=hits, inference_s=perf_counter() - start))
        arrays[f"{person}_second"] = data.ends[person] - 1
        arrays[person] = p
    return rows, arrays


def fit_neural(model, data, seed, selected, device):
    train = track.sample_clock(data, "fit", track.STEPS * track.BATCH, seed, True)
    validation, labels, _ = track.batch(data, None, data.roles["select"], selected, "Base", device)
    target = labels[:, -1].cpu().numpy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=track.LR)
    best, best_step, curve, elapsed = -1., 0, [], 0.
    for step in range(track.CHECK, track.STEPS + 1, track.CHECK):
        start = perf_counter()
        model.train()
        running = torch.zeros((), device=device)
        for i in range(step - track.CHECK, step):
            clocks = train[i * track.BATCH:(i + 1) * track.BATCH]
            x, y, _ = track.batch(data, None, data.roles["fit"], clocks, "Base", device)
            logits = model(x, None)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            running += loss.detach()
        loss = (running / track.CHECK).item()
        if not np.isfinite(loss):
            raise FloatingPointError(f"{model.kind} seed={seed} step={step}: nonfinite loss")
        elapsed += perf_counter() - start
        model.eval()
        score = track.macro_f1(track.confusion(target, track.probabilities(model, validation, None).argmax(1)))
        curve.append(dict(step=step, loss=loss, macro_f1=score, training_s=elapsed))
        if score > best:
            best, best_step = score, step
            saved = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(f"{model.kind} seed={seed} step={step} inner_macro={score:.4f} train_s={elapsed:.1f}", flush=True)
        if step - best_step >= track.CHECK * track.PATIENCE:
            break
    model.load_state_dict(saved)
    return dict(best_step=best_step, curve=curve, trainable=sum(p.numel() for p in model.parameters())), saved, train


def run(method, data, fold, seed, selected, device, db, source):
    previous = db.execute("SELECT receipt FROM runs WHERE method=? AND fold=? AND seed=?",
                          (method, fold, seed)).fetchone()
    if previous and json.loads(previous[0])["source"] == source:
        return
    torch.manual_seed(seed)
    started = perf_counter()
    if method in TABULAR:
        train = track.sample_clock(data, "fit", track.STEPS * track.BATCH, seed, True)
        x, y = sampled_features(data, "fit", train, method, True)
        v, vy = sampled_features(data, "select", selected, method, False)
        model = TabularBaseline(method, seed, device)
        training, weights = model.fit(x, y, v, vy)
        rows, arrays = tabular_nights(model, data, method)
    else:
        model = NeuralBaseline(method).to(device).eval()
        x, _, _ = track.batch(data, None, data.roles["select"], selected[:2], "Base", device)
        changed = x.clone()
        changed[..., -8:] += 7
        z = model(x, None)
        assert z.shape == (2, 5, 5) and z.isfinite().all()
        torch.testing.assert_close(z[:, :-1], model(changed, None)[:, :-1], rtol=0, atol=2e-6)
        with torch.no_grad():
            torch.testing.assert_close(z[:, -1], model(x, None)[:, -1], rtol=0, atol=2e-6)
        del z
        training, state, train = fit_neural(model, data, seed, selected, device)
        rows, arrays = track.score_nights(model, data, None, device)
        buffer = io.BytesIO()
        torch.save(state, buffer)
        weights = buffer.getvalue()
    receipt = dict(source=source, subjects=data.roles, training=training, results=rows, seconds=perf_counter() - started,
                   device=str(device), mean=data.mean.tolist(), scale=data.scale.tolist())
    buffer = io.BytesIO()
    np.savez_compressed(buffer, train_index=train, select_index=selected, **arrays)
    with db:
        db.execute("INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                   (method, fold, seed, json.dumps(receipt), weights, buffer.getvalue()))
    print(f"封存 {method} fold={fold} seed={seed} total_s={receipt['seconds']:.1f}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--folds", nargs="+", type=int, choices=(1, 2, 3), default=(1, 2, 3))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, choices=track.SEEDS, default=track.SEEDS)
    args = parser.parse_args(argv)
    from 发布复现 import require_data
    require_data()
    track.OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    paths = [Path(__file__), Path(track.__file__), track.SPLIT, track.MANIFEST,
             *sorted((LITERATURE / "methods").glob("*.py")),
             *(Path(m.__file__) for m in (cnn1d, inceptiontime, aix_tsdnet_3c)),
             LEGACY / "algorithms/_layers.py", track.BREATH / "src/classical.py",
             *(Path(p) for p in (track.FoldData.__init__.__code__.co_filename, track.totals.__code__.co_filename)),
             *(Path(cls.forward.__code__.co_filename) for cls in (CnnLstmTeacher, TCN, ResNet1D, YuLSTMCNN, ChengMIL))]
    source = {str(p): track.hash_file(p) for p in paths}
    roster = track.route(pd.read_csv(track.SPLIT), 1, (1, 2, 3))
    with threadpool_limits(limits=2), sqlite3.connect(track.OUTPUT / "60s五分类baseline.sqlite", timeout=120) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE IF NOT EXISTS runs (method TEXT, fold INTEGER, seed INTEGER, "
                   "receipt TEXT, weights BLOB, arrays BLOB, PRIMARY KEY(method, fold, seed))")
        for fold in args.folds:
            data = track.FoldData(1, roles=roster[fold])
            selected = track.sample_clock(data, "select", track.SELECT_SIZE, 74121, False)
            for method in args.methods:
                for seed in args.seeds:
                    run(method, data, fold, seed, selected, device, db, source)


if __name__ == "__main__":
    main()
