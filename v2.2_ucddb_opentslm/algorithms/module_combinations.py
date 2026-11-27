"""共用首层 OpenTSLM 与固定通道说明的组合模型。"""
import torch
from torch import nn
from torch.nn import functional as F

from algorithms.factored_readout import joint_log_probabilities, question_tokens
from algorithms.opentslm import GEMMA, RespirationOpenTSLM, load_checkpoint
from algorithms.synchronous_tokens import flatten_tokens

COMBINATIONS = {
    "baseline": (),
    "C01": ("M02",),
    "C02": ("M01",),
    "C03": ("M03",),
    "C04": ("M02", "M01"),
    "C05": ("M02", "M03"),
}


class ModuleCombination(RespirationOpenTSLM):
    def __init__(self, method, device):
        super().__init__(load_checkpoint(), device)
        self.modules_enabled = COMBINATIONS[method]
        self.register_buffer("fixed_prefix", torch.cat([
            getattr(self, f"channel_text_{i}") for i in range(4)
        ], dim=1), persistent=False)
        if "M02" in self.modules_enabled:
            self.amplitude_projection = nn.Parameter(torch.zeros_like(self.channel_embedding[:3]))
        if "M03" in self.modules_enabled:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(GEMMA, local_files_only=True)
            tokens, answers = question_tokens(tokenizer)
            with torch.no_grad():
                embeddings = self.llm.get_input_embeddings()
                self.register_buffer("questions", torch.cat([embeddings(t.to(device)) for t in tokens], dim=1),
                                     persistent=False)
                self.register_buffer("question_answers", answers.to(device), persistent=False)
            self.first_answer = tokens[0].shape[1] - 1

    def forward(self, features, amplitude):
        projected = self.projector(features.float()) + self.channel_embedding[:, None, :]
        if "M02" in self.modules_enabled:
            delta = amplitude[..., None] * self.amplitude_projection[:, None, :]
            projected = projected + F.pad(delta, (0, 0, 0, 0, 0, 1))
        signals = flatten_tokens(projected, "M01" in self.modules_enabled).to(self.fixed_prefix.dtype)
        factored = "M03" in self.modules_enabled
        prompt = self.questions if factored else self.prompt_0
        embeddings = torch.cat((self.fixed_prefix.expand(len(signals), -1, -1), signals,
                                prompt.expand(len(signals), -1, -1)), dim=1)
        base = self.llm.get_base_model()
        states = base.model(inputs_embeds=embeddings, use_cache=False).last_hidden_state
        output_weight = base.get_output_embeddings().weight
        if factored:
            positions = [self.fixed_prefix.shape[1] + signals.shape[1] + self.first_answer, -1]
            logits = torch.stack([
                F.linear(states[:, position], output_weight[answer_ids])
                for position, answer_ids in zip(positions, self.question_answers)
            ], dim=1).float()
        else:
            logits = F.linear(states[:, -1], output_weight[self.answer_0]).float()
        softcap = base.config.final_logit_softcapping
        if softcap:
            logits = (logits / softcap).tanh() * softcap
        return joint_log_probabilities(logits) if factored else F.log_softmax(logits, dim=-1)
