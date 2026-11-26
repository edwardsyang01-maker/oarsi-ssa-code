import torch
import torch.nn.functional as F
from loss_policy import loss_terms
from aggressive_policy import corn_terms, corn_log_probabilities, augment_aggressive

def log_probabilities(logits, plan):
    if plan['head_type'] == 'corn_conditional':
        return corn_log_probabilities(logits)
    assert plan['head_type'] == 'softmax5'
    return F.log_softmax(logits.float(), dim=1)

def training_augmentation(images, plan):
    from augmentation_policy import augment_knee
    return augment_knee(images, plan['augmentation_parameters'], enabled=plan['augment'])

def training_terms(logits, targets, plan, global_count, global_eligible=None):
    if plan['head_type'] == 'softmax5':
        terms = loss_terms(logits, targets, plan['class_weights'], plan['ordinal_lambda'])
        terms['predictions'] = logits.argmax(dim=1)
        return terms
    assert plan['head_type'] == 'corn_conditional'
    assert global_eligible is not None and global_eligible >= global_count > 0
    terms = corn_terms(logits, targets, plan['class_weights'], gamma=plan['focal_gamma'])
    logp = terms['log_probabilities']
    objective = (plan['focal_mix'] * terms['focal'] +
                 plan['corn_mix'] * terms['corn_sum'] * (global_count/global_eligible))
    # Caller sums each shard and multiplies by world_size/global_count:
    # after DS gradient averaging, exact global focal/N + conditional BCE/eligible.
    ce = -logp.gather(1, targets.long()[:, None]).squeeze(1)
    cdf = logp.exp().cumsum(dim=1)[:, :4]
    truth = torch.arange(4, device=targets.device)[None, :] >= targets[:, None]
    return {'objective': objective, 'cross_entropy': ce,
            'ordinal_rps': (cdf-truth.float()).square().mean(dim=1),
            'predictions': logp.argmax(dim=1)}

