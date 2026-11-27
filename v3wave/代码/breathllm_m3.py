"""BreathLLM-M3 的紧凑实现。

输入是 [batch, 4, samples]，通道顺序为 Flow、ribcage、abdo、SpO2，采样率 8 Hz。
模块只读取当前输入末端以前的数据。DTEMM 的 step 必须在每个新到达秒只调用一次。
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn


RATE = 8
SCALES = (5, 10, 20, 60)
CHANNELS = ("Flow", "ribcage", "abdo", "SpO2")
EPS = 1e-5


def _span(x: Tensor) -> Tensor:
    """稳健幅值跨度，最后一维是时间。"""
    return torch.quantile(x, .9, dim=-1) - torch.quantile(x, .1, dim=-1)


def _energy(x: Tensor) -> Tensor:
    """阻尼振荡的无量纲能量代理。"""
    velocity = torch.diff(x, dim=-1) * RATE
    omega = 2 * math.pi * .25
    return .5 * (velocity.square().mean(-1) + (omega * x[..., 1:]).square().mean(-1))


def _ar2(reference: Tensor) -> Tensor:
    """只用窗口较早部分拟合稳定 AR(2) 系数。"""
    left = reference[..., :-2]
    middle = reference[..., 1:-1]
    target = reference[..., 2:]
    design = torch.stack((middle, left), dim=-1)
    gram = design.transpose(-1, -2) @ design
    gram = gram + torch.eye(2, device=reference.device, dtype=reference.dtype) * 1e-3
    rhs = design.transpose(-1, -2) @ target.unsqueeze(-1)
    coefficient = torch.linalg.solve(gram, rhs).squeeze(-1)
    return coefficient.clamp(-1.8, 1.8)


@dataclass
class WDAMTOutput:
    values: Tensor       # [B, scale, channel, (r, e_perp, residual, quality)]
    valid: Tensor        # [B, scale, channel]
    summary: Tensor      # [B, channel, feature]


class WDAMT(nn.Module):
    """多尺度幅值与动力学证据。reference 必须是紧邻当前窗之前的 60 秒。"""

    def __init__(self):
        super().__init__()
        self.scale_gate = nn.Linear(4, 1)
        nn.init.zeros_(self.scale_gate.weight)
        nn.init.zeros_(self.scale_gate.bias)

    def forward(self, signal: Tensor, reference: Tensor) -> WDAMTOutput:
        if signal.ndim != 3 or signal.shape[1:] != (4, 60 * RATE):
            raise ValueError("signal 必须是 [batch, 4, 480] 的当前因果窗口")
        if reference.shape != signal.shape:
            raise ValueError("reference 必须是紧邻当前窗口之前的 [batch, 4, 480]")
        signal, reference = signal.float(), reference.float()
        blocks = []
        valid_blocks = []
        for seconds in SCALES:
            samples = seconds * RATE
            local = signal[..., -samples:]
            baseline_wave = reference[..., -samples:]
            centre = baseline_wave.median(dim=-1, keepdim=True).values
            local, baseline_wave = local - centre, baseline_wave - centre
            baseline_raw = _span(baseline_wave)
            baseline = baseline_raw.clamp_min(EPS)
            valid = (torch.isfinite(local).all(dim=-1) & torch.isfinite(baseline_wave).all(dim=-1)
                     & (baseline_raw > EPS))
            amplitude = _span(local).clamp_min(EPS)
            relative = (amplitude.log() - baseline.log()).clamp(-6, 4)
            local_energy = _energy(local).clamp_min(EPS)
            baseline_energy = _energy(baseline_wave).clamp_min(EPS)
            energy_ratio = (local_energy / baseline_energy).log()
            energy_ratio = energy_ratio.clamp(-8, 6)
            coefficient = _ar2(baseline_wave[:, :3])
            residual = (local[:, :3, 2:] - coefficient[..., 0, None] * local[:, :3, 1:-1]
                        - coefficient[..., 1, None] * local[:, :3, :-2])
            residual = residual.square().mean(-1).sqrt() / baseline[:, :3]
            residual = torch.cat((residual, torch.zeros_like(residual[:, :1])), dim=1)
            quality = (1 + residual).reciprocal()
            e_perp = energy_ratio - 2 * relative
            feature = torch.stack((relative, e_perp, residual, quality), dim=-1)
            feature[:, 3, 1:3] = 0
            feature[:, 3, 2] = 0
            feature[:, 3, 3] = 1
            feature = torch.nan_to_num(feature)
            blocks.append(feature)
            valid_blocks.append(valid)
        values = torch.stack(blocks, dim=1)
        valid = torch.stack(valid_blocks, dim=1)
        logits = self.scale_gate(values).squeeze(-1)
        logits = logits.masked_fill(~valid, -1e4)
        weights = logits.softmax(dim=1) * valid
        summary = (values * weights.unsqueeze(-1)).sum(dim=1)
        return WDAMTOutput(values, valid, summary)


@dataclass
class MCDPTOutput:
    values: Tensor       # [B, scale, (suppression, effort, contrast, central, obstructive, mismatch, corr, valid)]
    summary: Tensor      # [B, 8]
    mechanism: Tensor    # [B, 2], central and obstructive evidence


class MCDPT(nn.Module):
    """按时位保留气流、胸腹努力和能量关系。"""

    def __init__(self):
        super().__init__()
        self.scale_gate = nn.Linear(8, 1)
        nn.init.zeros_(self.scale_gate.weight)
        nn.init.zeros_(self.scale_gate.bias)

    def forward(self, signal: Tensor, dynamics: WDAMTOutput) -> MCDPTOutput:
        values = dynamics.values
        relative = values[..., 0]
        energy = values[..., 1]
        flow = relative[..., 0]
        effort = relative[..., 1:3].mean(-1)
        suppression = flow.neg().relu()
        contrast = effort - flow
        central = suppression * effort.neg().relu()
        obstructive = suppression * effort.relu()
        mismatch = energy[..., 1:3].mean(-1) - energy[..., 0]
        correlations = []
        for seconds in SCALES:
            tail = signal[..., -seconds * RATE:]
            chest = tail[:, 1] - tail[:, 1].mean(-1, keepdim=True)
            abdomen = tail[:, 2] - tail[:, 2].mean(-1, keepdim=True)
            correlations.append(torch.nn.functional.cosine_similarity(chest, abdomen, dim=-1))
        correlation = torch.stack(correlations, dim=1)
        valid = dynamics.valid[..., :3].all(-1).float()
        relation = torch.stack((suppression, effort, contrast, central, obstructive,
                                mismatch, correlation, valid), dim=-1)
        relation = torch.nan_to_num(relation)
        logits = self.scale_gate(relation).squeeze(-1).masked_fill(valid == 0, -1e4)
        weights = logits.softmax(dim=1) * valid
        summary = (relation * weights.unsqueeze(-1)).sum(dim=1)
        mechanism = summary[:, 3:5].clamp_min(0)
        return MCDPTOutput(relation, summary, mechanism)


@dataclass
class MemoryState:
    central: Tensor
    obstructive: Tensor
    central_mass: Tensor
    obstructive_mass: Tensor
    central_to_obstructive: Tensor
    obstructive_to_central: Tensor


class DTEMM(nn.Module):
    """每秒只写入新到达 1 秒的双机制记忆。"""

    def __init__(self, width: int = 32, tau: float = 20., gamma: float = .25):
        super().__init__()
        self.width = width
        self.tau = tau
        self.gamma = gamma
        self.new_block = nn.Linear(4 * RATE, width)
        self.readout = nn.Linear(2 * width + 1, width, bias=False)
        nn.init.zeros_(self.readout.weight)

    def initial(self, batch: int, device, dtype=torch.float32) -> MemoryState:
        zero = lambda *shape: torch.zeros(batch, *shape, device=device, dtype=dtype)
        return MemoryState(zero(self.width), zero(self.width), zero(1), zero(1), zero(1), zero(1))

    def new_state(self, signal: Tensor, reference: Tensor) -> Tensor:
        if signal.shape[-1] < RATE:
            raise ValueError("每次记忆更新至少需要最新1秒的8个采样点")
        centre = reference.median(dim=-1, keepdim=True).values
        scale = _span(reference).clamp_min(EPS).unsqueeze(-1)
        increment = (signal[..., -RATE:] - centre) / scale
        return self.new_block(increment.flatten(1))

    def step(self, state: MemoryState, new_state: Tensor, mechanism: Tensor,
             pause_probability: Tensor | None = None) -> MemoryState:
        if mechanism.shape[-1] != 2:
            raise ValueError("mechanism 必须是 [batch, 2] 的中枢和阻塞证据")
        central_weight, obstructive_weight = mechanism.unbind(-1)
        central_weight = central_weight.clamp_min(0)
        obstructive_weight = obstructive_weight.clamp_min(0)
        if pause_probability is None:
            pause_probability = 1 - torch.exp(-(central_weight + obstructive_weight))
        rho = math.exp(-1 / self.tau) * (1 - self.gamma * (1 - pause_probability))
        rho = rho.clamp(0, 1).unsqueeze(-1)
        central_mass = rho * state.central_mass + central_weight.unsqueeze(-1)
        obstructive_mass = rho * state.obstructive_mass + obstructive_weight.unsqueeze(-1)
        central = rho * state.central + central_weight.unsqueeze(-1) * new_state
        obstructive = rho * state.obstructive + obstructive_weight.unsqueeze(-1) * new_state
        c_to_o = rho * state.central_to_obstructive + state.central_mass * obstructive_weight.unsqueeze(-1)
        o_to_c = rho * state.obstructive_to_central + state.obstructive_mass * central_weight.unsqueeze(-1)
        return MemoryState(central, obstructive, central_mass, obstructive_mass, c_to_o, o_to_c)

    def read(self, state: MemoryState) -> Tensor:
        direction = ((state.central_to_obstructive - state.obstructive_to_central)
                     / (state.central_to_obstructive + state.obstructive_to_central + EPS))
        central = state.central / state.central_mass.clamp_min(EPS)
        obstructive = state.obstructive / state.obstructive_mass.clamp_min(EPS)
        return self.readout(torch.cat((central, obstructive, direction), dim=-1))

    @staticmethod
    def detach(state: MemoryState) -> MemoryState:
        return MemoryState(*(getattr(state, name).detach() for name in state.__dataclass_fields__))


@dataclass
class M3Evidence:
    dynamics: WDAMTOutput
    relations: MCDPTOutput
    memory: Tensor
    state: MemoryState

    @property
    def vector(self) -> Tensor:
        return torch.cat((self.dynamics.summary.flatten(1), self.relations.summary, self.memory), dim=-1)


class M3EvidenceModel(nn.Module):
    """三个证据模块的统一入口，不包含 OpenTSLM 主干。"""

    def __init__(self, memory_width: int = 32):
        super().__init__()
        self.dynamics = WDAMT()
        self.relations = MCDPT()
        self.memory = DTEMM(memory_width)

    def forward(self, signal: Tensor, reference: Tensor, state: MemoryState | None = None,
                update_memory: bool = False) -> M3Evidence:
        dynamics = self.dynamics(signal, reference)
        relations = self.relations(signal, dynamics)
        state = self.memory.initial(len(signal), signal.device, signal.dtype) if state is None else state
        if update_memory:
            state = self.memory.step(state, self.memory.new_state(signal, reference), relations.mechanism)
        memory = self.memory.read(state)
        return M3Evidence(dynamics, relations, memory, state)


class SubtypeResidual(nn.Module):
    """零初始化的条件亚型读出，只改 OSA、CSA、MSA 的内部比例。"""

    def __init__(self, evidence_width: int, classes: int = 5):
        super().__init__()
        self.projection = nn.Linear(evidence_width, 3)
        self.classes = classes
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, base_log_prob: Tensor, evidence: M3Evidence) -> Tensor:
        if base_log_prob.shape[-1] != self.classes:
            raise ValueError("base_log_prob 的最后一维必须是五分类")
        subtype = self.projection(evidence.vector).to(base_log_prob.dtype)
        while subtype.ndim < base_log_prob.ndim:
            subtype = subtype.unsqueeze(1)
        probability = base_log_prob.softmax(-1)
        subtype_mass = probability[..., 1:4].sum(-1, keepdim=True)
        subtype_probability = probability[..., 1:4] / subtype_mass.clamp_min(EPS)
        subtype_probability = (subtype_probability.clamp_min(EPS).log() + subtype).softmax(-1)
        result = probability.clone()
        result[..., 1:4] = subtype_mass * subtype_probability
        return result.clamp_min(EPS).log()


class M3SoftPrompt(nn.Module):
    """把证据加到亚型隐藏状态，不改异常检测和暂停程度分支。"""

    def __init__(self, evidence_width: int, hidden_width: int):
        super().__init__()
        self.projection = nn.Linear(evidence_width, hidden_width, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, subtype_state: Tensor, evidence: M3Evidence) -> Tensor:
        if subtype_state.ndim != 3 or subtype_state.shape[1] != 5:
            raise ValueError("subtype_state 应为五个秒级输出的 [batch, 5, hidden] 张量")
        result = subtype_state.clone()
        result[:, -1] = result[:, -1] + self.projection(evidence.vector).to(result.dtype)
        return result


def track_a_evidence(history: Tensor) -> tuple[Tensor, Tensor]:
    """返回截至当前秒的三个相对尺度及各尺度的同期通道证据。"""
    reference = history[..., -55 * RATE:-35 * RATE]
    centre = reference.mean(-1, keepdim=True)
    reference = reference - centre
    span = _span(reference)
    baseline = span.clamp_min(1e-4)
    energy = _energy(reference[:, :3])
    coefficients = _ar2(reference[:, :3] / baseline[:, :3, None])
    dynamics, relations = [], []
    for seconds in SCALES[:-1]:
        local = history[..., -seconds * RATE:]
        centered = local - local.mean(-1, keepdim=True)
        amplitude = _span(centered).clamp_min(1e-4)
        ratio = amplitude.log() - baseline.log()
        power = _energy(centered[:, :3]).clamp_min(EPS)
        log_energy = power.log() - energy.clamp_min(EPS).log()
        shape_energy = (log_energy - 2 * ratio[:, :3]).clamp(-8, 8)
        normalised = centered[:, :3] / baseline[:, :3, None]
        residual = normalised[..., 2:] - coefficients[..., 0, None] * normalised[..., 1:-1]
        residual = (residual - coefficients[..., 1, None] * normalised[..., :-2]).square().mean(-1).sqrt()
        residual = torch.nn.functional.pad(residual.clamp_max(50), (0, 1))
        valid = span.gt(1e-4)
        ratio = ratio.clamp(-6, 4)
        values = torch.stack((ratio, torch.nn.functional.pad(shape_energy, (0, 1)),
                              residual.log1p(), (1 + residual).reciprocal()), -1)
        dynamics.append(values * valid[..., None])

        flow, chest, abdomen = ratio[:, :3].unbind(-1)
        suppression = ((-flow - .25) * 3).sigmoid()
        effort = (chest + abdomen) * .5
        effort_present = ((effort + .5) * 4).sigmoid()
        corr = torch.nn.functional.cosine_similarity(centered[:, 1], centered[:, 2], dim=-1)
        ok = valid[:, :3].all(-1)
        relation = torch.stack((suppression, effort, effort - flow,
                                suppression * (1 - effort_present), suppression * effort_present,
                                (log_energy[:, 1:].mean(-1) - log_energy[:, 0]).clamp(-8, 8),
                                shape_energy[:, 1:].mean(-1) - shape_energy[:, 0],
                                corr, ok), -1)
        relations.append(relation * ok[:, None])
    return torch.stack(dynamics, 1), torch.stack(relations, 1)


class DynamicsPrompt(nn.Module):
    """按尺度融合波形和动力学，给每个末端信号词元添加残差。"""

    def __init__(self, hidden: int):
        super().__init__()
        self.values = nn.Linear(16, 24)
        self.gate = nn.Linear(16, 1)
        self.wave = nn.Linear(hidden, 24, bias=False)
        self.output = nn.Linear(24, hidden, bias=False)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.output.weight)

    def forward(self, local: Tensor, dynamics: Tensor, signal: Tensor) -> Tensor:
        context = []
        for stop in range(56 * RATE, 61 * RATE, RATE):
            wave = signal[..., :stop]
            wave = wave - wave.mean(-1, keepdim=True)
            amplitude = wave.square().mean(-1).sqrt().log1p()
            oscillation = torch.nn.functional.pad(_energy(wave[:, :3]).log1p(), (0, 1))
            zero = torch.zeros_like(amplitude)
            context.append(torch.stack((amplitude, oscillation, zero, zero), -1))
        context = torch.stack(context, 1).unsqueeze(2)
        features = torch.cat((dynamics, context), dim=2).flatten(-2)
        values = self.values(features) + self.wave(local.float()).unsqueeze(2)
        values = torch.nn.functional.gelu(values)
        return self.output((values * self.gate(features).softmax(dim=2)).sum(2)).to(local.dtype)


class ConditionalEvidence(nn.Module):
    """按时位保留关系证据或记忆，仅向暂停亚型读出提供残差。"""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.evidence = nn.Linear(dim, 32)
        self.query = nn.Linear(hidden, 32, bias=False)
        self.output = nn.Linear(32, 3, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, state: Tensor, evidence: Tensor) -> Tensor:
        return self.output(torch.nn.functional.gelu(self.query(state) + self.evidence(evidence)))


def _smoke() -> None:
    torch.manual_seed(4121)
    model = M3EvidenceModel()
    signal = torch.randn(2, 4, 480)
    reference = torch.randn_like(signal)
    evidence = model(signal, reference)
    assert evidence.vector.shape == (2, 4 * 4 + 8 + 32)
    state = model.memory.initial(2, signal.device, signal.dtype)
    for second in range(3):
        current = torch.randn_like(signal)
        evidence = model(current, reference, state, update_memory=True)
        state = evidence.state
    assert torch.isfinite(evidence.vector).all()
    assert not torch.equal(state.central, model.memory.initial(2, signal.device, signal.dtype).central)
    extended = torch.cat((torch.randn(2, 4, 40), signal), dim=-1)[..., -480:]
    torch.testing.assert_close(model(extended, reference).vector, model(signal, reference).vector)
    base = torch.randn(2, 5).log_softmax(-1)
    head = SubtypeResidual(evidence.vector.shape[-1])
    refined = head(base, evidence)
    torch.testing.assert_close(refined, base)
    torch.testing.assert_close(refined[:, [0, 4]], base[:, [0, 4]])
    print("smoke ok", tuple(evidence.vector.shape), "memory_norm=", round(state.central.norm().item(), 5))


def _preflight(subject: str) -> None:
    project = Path(__file__).resolve().parents[2]
    breath5 = project / "breath5"
    sys.path.insert(0, str(breath5))
    from src.experiment_data import PSG, waveform

    raw = torch.from_numpy(waveform(project / "data/ucddb" / f"{subject}.rec", PSG))
    raw = torch.nan_to_num(raw)
    model = M3EvidenceModel().eval()
    first_end = 120
    if raw.shape[-1] // RATE <= first_end + 4:
        raise ValueError(f"记录太短，无法构造因果前后窗口：{subject}")
    state = model.memory.initial(1, raw.device)
    with torch.inference_mode():
        for end in range(first_end, first_end + 5):
            reference = raw[:, (end - 120) * RATE:(end - 60) * RATE].unsqueeze(0)
            signal = raw[:, (end - 60) * RATE:end * RATE].unsqueeze(0)
            output = model(signal, reference, state, update_memory=True)
            state = output.state
            if not torch.isfinite(output.vector).all():
                raise FloatingPointError(f"非有限特征：{subject} second={end}")
    print(f"preflight ok subject={subject} seconds={first_end}-{first_end + 4} "
          f"feature_width={output.vector.shape[-1]} memory_norm={state.central.norm().item():.5f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="运行因果形状、分支和记忆探针")
    parser.add_argument("--preflight", metavar="SUBJECT", help="用一份 UCDDB 记录验证输入与一秒写入")
    args = parser.parse_args()
    if args.smoke:
        _smoke()
    elif args.preflight:
        _preflight(args.preflight)
    else:
        parser.error("请指定 --smoke 或 --preflight SUBJECT")


if __name__ == "__main__":
    main()
