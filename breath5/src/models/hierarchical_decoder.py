import torch
from torch import nn
from torch.nn import functional as F


def five_class_log_probs(event, apnea, subtype):
    """Return log probabilities in Normal, OSA, CSA, MSA, Hypopnea order."""
    p_event = F.logsigmoid(event.float())
    p_apnea = F.logsigmoid(apnea.float())
    p_subtypes = p_event[..., None] + p_apnea[..., None] + subtype.float().log_softmax(-1)
    normal = F.logsigmoid(-event.float())
    hypopnea = p_event + F.logsigmoid(-apnea.float())
    return torch.cat((normal[..., None], p_subtypes, hypopnea[..., None]), dim=-1)


def project_probabilities(p):
    n, o, c, m, h = p.unbind(-1)
    return torch.stack((n, h, o, c + m), -1), torch.stack((n, o + c + m + h), -1)


class HierarchicalDecoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.event = nn.Linear(width, 1)
        self.apnea = nn.Linear(width, 1)
        self.subtype = nn.Linear(width, 3)

    def forward(self, event_state, subtype_state):
        return self.event(event_state).squeeze(-1), self.apnea(event_state).squeeze(-1), self.subtype(subtype_state)
