"""Causal five-class adaptation of the existing two-question OpenTSLM."""

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from .hierarchical_decoder import HierarchicalDecoder, five_class_log_probs


@dataclass
class ForwardResult:
    log_prob: torch.Tensor
    conditional_logits: tuple
    boundary_logits: torch.Tensor


class E0FiveClass(nn.Module):
    def __init__(self, backbone, fusion_weight=0.5, compact=False):
        super().__init__()
        self.backbone = backbone
        self.fusion_weight = fusion_weight
        width = backbone.questions.size(-1)
        self.tail_projection = nn.Linear(8, width)
        self.time_positions = nn.Parameter(torch.zeros(16, width))
        self.decoder = HierarchicalDecoder(width)
        self.physiology = HierarchicalDecoder(10)
        self.boundary = nn.Linear(width, 3)
        self.compact = compact
        if compact:
            first = backbone.questions[:, :backbone.first_answer+1]
            second = backbone.questions[:, backbone.first_answer+1:]
            questions = torch.stack([(p.mean(1) + p[:, -1]) * .5 for p in (first, second)], dim=1)
            self.register_buffer("compact_questions", questions, persistent=False)
            self.history_mix = nn.Parameter(torch.ones(4, width))
            self.tail_mix = nn.Parameter(torch.ones(4, width))
        self._initialize_from_answers()

    @classmethod
    def from_existing(cls, project: Path, device, fusion_weight=0.5, compact=False):
        sys.path.insert(0, str(project / "v2.2_ucddb_opentslm"))
        from algorithms.question_evidence_prompt import QuestionEvidencePrompt

        return cls(QuestionEvidencePrompt(device), fusion_weight=fusion_weight, compact=compact).to(device)

    @torch.no_grad()
    def _initialize_from_answers(self):
        backbone = self.backbone
        unembedding = backbone.llm.get_base_model().get_output_embeddings().weight
        event_rows, subtype_rows = unembedding[backbone.question_answers].float()
        self.decoder.event.weight.copy_(event_rows[1] - event_rows[0])
        self.decoder.event.bias.zero_()
        self.decoder.subtype.weight[:2].copy_(subtype_rows.flip(0))
        self.decoder.subtype.bias[:2].zero_()

    @torch.no_grad()
    def _encode_early(self, signal):
        encoder = self.backbone.encoder
        batch, channels, samples = signal.shape
        x = encoder.patch_embed(signal.reshape(batch * channels, 1, samples)).transpose(1, 2)
        x = encoder.input_norm(x + encoder.pos_embed[:, :x.shape[1]])
        x = encoder.encoder.layers[0](x)
        return x.reshape(batch, channels, samples // 40, 10, -1).mean(dim=3)

    def fold_local_features(self, signal):
        """Expose reusable, label-free tokens for each disjoint history block."""
        return self._encode_early(signal[..., :240]), self._encode_early(signal[..., 240:440])

    def encode_states(self, signal, history_tokens=None):
        if signal.ndim != 3 or signal.shape[1:] != (4, 480):
            raise ValueError("Expected [batch, 4, 480] with causal 8 Hz PSG")
        base = self.backbone
        early = torch.cat(self.fold_local_features(signal) if history_tokens is None else history_tokens, dim=2)
        pos = self.time_positions
        early = base.projector(early.float()) + base.channel_embedding[None, :, None] + pos[:11]
        if self.compact:
            early = (early * self.history_mix[None, :, None]).mean(1)
        else:
            early = early.transpose(1, 2).flatten(1, 2)

        tail = signal[..., 440:].reshape(-1, 4, 5, 8).transpose(1, 2)
        scaling = signal[:, :3, :440].square().mean(-1).add(1e-6).sqrt()
        rms = tail[:, :, :3].square().mean(-1).add(1e-6).sqrt()
        ratio = (rms.log() - scaling[:, None, :].log()).clamp(-5, 5)
        tail_embeddings = self.tail_projection(tail.float()) + base.channel_embedding + pos[11:, None]
        amplitude = ratio[..., None] * base.amplitude_projection
        tail_embeddings = tail_embeddings + F.pad(amplitude, (0, 0, 0, 1))

        effort = (ratio[..., 1] + ratio[..., 2]) * .5 - ratio[..., 0]
        common = tail_embeddings[:, :, :3].mean(dim=2)
        question_bias = torch.stack((common + amplitude[:, :, 0], common + amplitude[:, :, 1:3].mean(2)), dim=2)
        prefix = base.fixed_prefix.expand(len(signal), -1, -1)
        if self.compact:
            questions = self.compact_questions[:, None] + question_bias
            local = (tail_embeddings * self.tail_mix).mean(2, keepdim=True)
            groups = torch.cat((local, questions), dim=2)
            prefix = prefix.mean(1, keepdim=True)
            event_position = 1
        else:
            questions = base.questions[:, None] + torch.einsum("pl,bsph->bslh", base.answer_mask, question_bias)
            groups = torch.cat((tail_embeddings, questions), dim=2)
            event_position = 4 + base.first_answer
        embeddings = torch.cat((prefix, early.to(prefix.dtype), groups.flatten(1, 2).to(prefix.dtype)), dim=1)
        states = base.llm.get_base_model().model(inputs_embeds=embeddings, use_cache=False).last_hidden_state
        states = states[:, -(5 * groups.shape[2]):].reshape(len(signal), 5, groups.shape[2], -1).float()
        event_state = states[:, :, event_position]
        subtype_state = states[:, :, -1]

        oxygen = tail[:, :, 3]
        centered_effort = tail[:, :, 1:3] - tail[:, :, 1:3].mean(-1, keepdim=True)
        correlation = F.cosine_similarity(centered_effort[:, :, 0], centered_effort[:, :, 1], dim=-1)
        evidence = torch.cat((
            ratio, effort[..., None], correlation[..., None],
            oxygen.mean(-1, keepdim=True),
            (oxygen[..., -1] - oxygen[..., 0])[..., None],
            oxygen.std(-1, unbiased=False, keepdim=True),
            (tail[:, :, :2, -1] - tail[:, :, :2, 0]),
        ), -1)
        return event_state, subtype_state, evidence

    def forward(self, signal, history_tokens=None):
        event_state, subtype_state, evidence = self.encode_states(signal, history_tokens)
        primary = self.decoder(event_state, subtype_state)
        expert = self.physiology(evidence, evidence)
        weight = self.fusion_weight
        logits = tuple((1 - weight) * a + weight * b for a, b in zip(primary, expert))
        return ForwardResult(five_class_log_probs(*logits), logits, self.boundary(event_state))
