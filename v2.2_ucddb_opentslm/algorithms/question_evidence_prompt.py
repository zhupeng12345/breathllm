"""向 OpenTSLM 的两道问题注入各自的窗内变化向量。"""

import torch
from torch.nn import functional as F

from algorithms.factored_readout import joint_log_probabilities
from algorithms.module_combinations import ModuleCombination


class QuestionEvidencePrompt(ModuleCombination):
    def __init__(self, device):
        super().__init__("C05", device)
        length = self.questions.shape[1]
        positions = torch.arange(length, device=device)
        self.register_buffer("answer_mask", torch.stack(
            [positions == self.first_answer, positions == length - 1]).float(), persistent=False)
        self.mode = "routed"
        self.region = "center"
        self.gain = 0.0

    def _evidence(self, projected, amplitude):
        if self.region == "center":
            local = projected[:, :, 2:4].mean(dim=2)
        elif self.region == "left":
            local = projected[:, :, :2].mean(dim=2)
        else:
            index = (amplitude[:, 0, :-1] + amplitude[:, 0, 1:]).argmin(dim=1)
            pairs = (projected[:, :, :-1] + projected[:, :, 1:]) * .5
            batch = torch.arange(len(projected), device=projected.device)
            local = pairs.transpose(1, 2)[batch, index]
        context = projected.mean(dim=2)
        delta = (local - context) * 1.5
        scale = projected.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().flatten(1).clamp_min(1e-3)
        delta = (delta / scale[:, :, None]).clamp(-3, 3)
        airflow = delta[:, 0]
        effort = (delta[:, 1] + delta[:, 2]) * .5 - airflow
        if self.mode == "routed":
            routes = (airflow, effort)
        elif self.mode == "swapped":
            routes = (effort, airflow)
        elif self.mode == "shared":
            summary = (airflow + effort) * .5
            routes = (summary, summary)
        else:
            routes = (torch.zeros_like(airflow), effort)
        return torch.stack(routes, dim=1)

    def forward(self, features, amplitude):
        if self.gain == 0:
            return super().forward(features, amplitude)

        projected = self.projector(features.float()) + self.channel_embedding[:, None, :]
        delta = amplitude[..., None] * self.amplitude_projection[:, None, :]
        projected = projected + F.pad(delta, (0, 0, 0, 0, 0, 1))
        cue = self._evidence(projected, amplitude)
        prompt_scale = self.questions.float().square().mean().sqrt()
        bias = torch.einsum("pt,bph->bth", self.answer_mask, cue)
        questions = (self.questions + bias * (self.gain * prompt_scale)).to(self.questions.dtype)
        prefix = self.fixed_prefix.expand(len(features), -1, -1)
        signal = projected.flatten(1, 2).to(prefix.dtype)
        embeddings = torch.cat((prefix, signal, questions), dim=1)
        base = self.llm.get_base_model()
        states = base.model(inputs_embeds=embeddings, use_cache=False).last_hidden_state
        positions = (prefix.shape[1] + signal.shape[1] + self.first_answer, -1)
        output_weight = base.get_output_embeddings().weight
        scores = torch.stack([
            F.linear(states[:, position], output_weight[answer_ids])
            for position, answer_ids in zip(positions, self.question_answers)
        ], dim=1).float()
        softcap = base.config.final_logit_softcapping
        if softcap:
            scores = (scores / softcap).tanh() * softcap
        return joint_log_probabilities(scores)
