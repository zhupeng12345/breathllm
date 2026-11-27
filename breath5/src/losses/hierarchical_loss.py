import torch
from torch.nn import functional as F


def _mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def classification_loss(output, labels, weights=None, teacher_logits=None, *,
                        hierarchy_weight=.3, distillation_weight=.05, temperature=2.):
    valid = labels >= 0
    weights = valid.float() if weights is None else weights * valid
    safe = labels.clamp_min(0)
    event, apnea, subtype = output.conditional_logits
    nll = -output.log_prob.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    base = _mean(nll, weights)
    true_event = labels > 0
    true_apnea = (labels >= 1) & (labels <= 3)
    hierarchy = _mean(F.binary_cross_entropy_with_logits(event, true_event.float(), reduction="none"), weights)
    hierarchy += _mean(F.binary_cross_entropy_with_logits(apnea, true_apnea.float(), reduction="none"),
                       weights * true_event)
    subclass = F.cross_entropy(subtype.flatten(0, -2), (labels - 1).clamp(0, 2).flatten(),
                               reduction="none").reshape_as(labels)
    hierarchy += _mean(subclass, weights * true_apnea)
    loss = base + hierarchy_weight * hierarchy
    if teacher_logits is not None:
        teacher = (teacher_logits.detach() / temperature).softmax(-1)
        student = (output.log_prob / temperature).log_softmax(-1)
        distill = F.kl_div(student, teacher, reduction="none").sum(-1)
        loss = loss + distillation_weight * temperature ** 2 * _mean(distill, weights)
    return loss
