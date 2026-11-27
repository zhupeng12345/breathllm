"""UCDDB 四通道软提示适配，共用作者的睡眠模型并约束答案词元。"""
import hashlib
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from breathllm_paths import WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = WEIGHTS / "gemma-3-270m-sleep-sp/model_checkpoint.pt"
GEMMA = WEIGHTS / "gemma-3-270m"
sys.path.insert(0, str(ROOT / "third_party/OpenTSLM/src"))
from opentslm.model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
from opentslm.model.projector.MLPProjector import MLPProjector


def load_checkpoint():
    with CHECKPOINT.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != "0bddf130cf5ec32311fa025a3228ebee58f205adb08337ed519c6cd62e96b432":
        raise ValueError("OpenTSLM 睡眠检查点与发布者的 SHA256 不符")
    return torch.load(CHECKPOINT, weights_only=False, map_location="cpu")


class SignalEncoder(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.encoder = TransformerCNNEncoder()
        self.encoder.load_state_dict(checkpoint["encoder_state"], strict=True)
        self.encoder.requires_grad_(False).eval()

    @torch.no_grad()
    def encode(self, signal):
        if signal.ndim != 3 or signal.shape[1:] != (4, 240):
            raise ValueError("输入形状应为 [批大小, 4, 240]")
        patches = self.encoder(signal.float().reshape(-1, 240))
        return patches.reshape(signal.size(0), 4, 6, 10, 128).mean(dim=3)


def load_encoder(device):
    return SignalEncoder(load_checkpoint()).to(device).eval()


class RespirationOpenTSLM(SignalEncoder):
    prompt_text = (
        "Classify this 30-second sleep interval. A: normal, B: central apnea, C: obstructive apnea. Answer:",
        "Classify the subtype of this hypopnea. B: central, C: obstructive. Answer:",
    )
    channel_names = ("Airflow:", "Chest belt:", "Abdominal belt:", "Oxygen saturation:")

    def __init__(self, checkpoint, device):
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        super().__init__(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(GEMMA, local_files_only=True)
        dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
        base_llm = AutoModelForCausalLM.from_pretrained(
            GEMMA, local_files_only=True, dtype=dtype, attn_implementation="sdpa",
        )
        if not checkpoint["lora_enabled"]:
            raise ValueError("睡眠检查点缺少预期的 LoRA")
        self.llm = get_peft_model(base_llm, checkpoint["lora_config"]["default"])
        lora_params = {k: p for k, p in self.llm.named_parameters() if "lora_" in k}
        if lora_params.keys() != checkpoint["lora_state"].keys():
            raise ValueError("语言模型 LoRA 参数的名称或数量不匹配")
        with torch.no_grad():
            for key, p in lora_params.items():
                source = checkpoint["lora_state"][key]
                if p.shape != source.shape:
                    raise ValueError(f"LoRA 权重维度不匹配 {key}")
                p.copy_(source)
        self.llm.requires_grad_(False).eval()

        hidden = base_llm.config.hidden_size
        self.projector = MLPProjector(128, hidden, device="cpu")
        self.projector.load_state_dict(checkpoint["projector_state"], strict=True)
        self.channel_embedding = nn.Parameter(torch.zeros(4, hidden))
        embed = self.llm.get_input_embeddings()
        with torch.no_grad():
            for i, label in enumerate(self.channel_names):
                ids = tokenizer.encode(label, add_special_tokens=(i == 0), return_tensors="pt")
                self.register_buffer(f"channel_text_{i}", embed(ids), persistent=False)
            for i, prompt in enumerate(self.prompt_text):
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                answer_ids = []
                for letter in ("A", "B", "C") if i == 0 else ("B", "C"):
                    complete = tokenizer.encode(prompt + " " + letter, add_special_tokens=False)
                    if complete[:-1] != ids:
                        raise ValueError(f"答案 {letter} 和提示无法分成独立词元")
                    answer_ids.append(complete[-1])
                if len(set(answer_ids)) != len(answer_ids):
                    raise ValueError("候选答案必须对应不同词元")
                self.register_buffer(f"prompt_{i}", embed(torch.tensor([ids])), persistent=False)
                self.register_buffer(f"answer_{i}", torch.tensor(answer_ids), persistent=False)
        self.to(device)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.llm.eval()
        return self

    def forward(self, features, auxiliary=False):
        if features.ndim != 4 or features.shape[1:] != (4, 6, 128):
            raise ValueError("编码特征应为 [批大小, 4, 6, 128]")
        chunks = []
        projected = self.projector(features.float()) + self.channel_embedding[:, None, :]
        for i in range(4):
            channel_text = getattr(self, f"channel_text_{i}")
            chunks.append(channel_text.expand(len(features), -1, -1))
            chunks.append(projected[:, i].to(channel_text.dtype))
        prompt = getattr(self, f"prompt_{int(auxiliary)}")
        chunks.append(prompt.expand(len(features), -1, -1))
        embeddings = torch.cat(chunks, dim=1)
        ids = getattr(self, f"answer_{int(auxiliary)}")
        base = self.llm.get_base_model()
        last = base.model(inputs_embeds=embeddings, use_cache=False).last_hidden_state[:, -1]
        logits = F.linear(last, base.get_output_embeddings().weight.index_select(0, ids))
        softcap = base.config.final_logit_softcapping
        if softcap:
            logits = (logits / softcap).tanh() * softcap
        return logits.float()


def load_model(device):
    return RespirationOpenTSLM(load_checkpoint(), device)
