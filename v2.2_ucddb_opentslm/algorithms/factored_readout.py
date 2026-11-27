"""M03 两个答案位置共同给出三类对数概率。"""
import torch
from torch.nn import functional as F

QUESTIONS = (
    " Is there an apnea in this interval? A: normal, B: apnea. Answer:",
    "\n If there is an apnea, what is its type? B: central, C: obstructive. Answer:",
)


def question_tokens(tokenizer):
    tokens, answers = [], []
    for prompt, letters in zip(QUESTIONS, ("AB", "BC")):
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = []
        for letter in letters:
            full = tokenizer.encode(prompt + " " + letter, add_special_tokens=False)
            if len(full) != len(ids) + 1 or full[:-1] != ids:
                raise ValueError("两问读出的答案需要独立词元")
            answer_ids.append(full[-1])
        tokens.append(torch.tensor([ids]))
        answers.append(answer_ids)
    return tokens, torch.tensor(answers)


def joint_log_probabilities(logits):
    detection = F.log_softmax(logits[:, 0].float(), dim=-1)
    subtype = F.log_softmax(logits[:, 1].float(), dim=-1)
    return torch.cat((detection[:, :1], detection[:, 1:] + subtype), dim=-1)
