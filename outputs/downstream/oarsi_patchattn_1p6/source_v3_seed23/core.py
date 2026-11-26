from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

TARGETS = ['osfm', 'osfl', 'ostm', 'ostl', 'jsm', 'jsl']
SWAP = [1, 0, 3, 2, 5, 4]  # medial <-> lateral: femoral and tibial osteophytes and JSN are swapped pairwise
GRID = 28
PATCHES = GRID*GRID
REGISTERS = 4
TOKENS = 1+REGISTERS+PATCHES
GROUPS = {
    # pool: patch attention pooling or not; oarsi: None / 'cls' / 'pool'; swap: swap medial/lateral targets on a random flip
    'P0': {'pool': False, 'oarsi': None, 'swap': True},
    'P1': {'pool': True, 'oarsi': None, 'swap': True},
    'P2': {'pool': False, 'oarsi': 'cls', 'swap': True},
    'P3': {'pool': True, 'oarsi': 'pool', 'swap': True},
    'P4': {'pool': True, 'oarsi': 'pool', 'swap': False},
}


def lambda_at(value, epoch):
    return value*([0.25, 0.5, 0.75, 1.0][epoch-1] if epoch <= 4 else 1.0)


def load_targets(rows, path, expected_sha=None):
    path = Path(path)
    if expected_sha is not None:
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        assert h == expected_sha, 'OARSI_LABEL_IDENTITY_CHANGED'
    with path.open() as stream:
        labels = [json.loads(line) for line in stream]
    assert len(labels) == len(rows), 'OARSI_LABEL_COUNT_MISMATCH'
    out = np.full((len(rows), len(TARGETS)), -1, dtype=np.int64)
    for i, (row, lab) in enumerate(zip(rows, labels)):
        assert lab['key'] == row['key'] and lab['split'] == row['split'] and int(lab['kl']) == int(row['kl']), 'OARSI_ROW_MISMATCH'
        assert lab['clinical_side'] == row['clinical_side'] and lab['visit'] == row['visit']
        for j, name in enumerate(TARGETS):
            v = lab[name]
            if v is not None:
                assert v in (0, 1, 2, 3), 'OARSI_GRADE_OUT_OF_RANGE'
                out[i, j] = v
    return out


def canonical_flip(images, is_left):
    assert images.ndim == 4 and is_left.shape == (images.shape[0],)
    mask = is_left.to(images.device).view(-1, 1, 1, 1)
    return torch.where(mask, images.flip(-1), images)


def random_flip(images, targets, mask, swap):
    flipped = torch.where(mask.view(-1, 1, 1, 1), images.flip(-1), images)
    if not swap:
        return flipped, targets
    swapped = targets[:, SWAP]
    return flipped, torch.where(mask.view(-1, 1).to(targets.device), swapped, targets)


def oarsi_loss(logits, targets):
    logits = logits.float()
    present = targets >= 0
    safe = targets.clamp_min(0)
    ce = F.cross_entropy(logits.reshape(-1, 4), safe.reshape(-1), reduction='none').view(targets.shape)
    ce = ce*present
    return ce.sum(1)/present.sum(1).clamp_min(1)


class PatchPool(nn.Module):
    def __init__(self, queries, dim=1024, heads=8, seed=17):
        super().__init__()
        self.k = queries
        base = int(seed)*10000
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(base+301)
            self.norm = nn.LayerNorm(dim)
            self.position = nn.Parameter(torch.empty(PATCHES, dim))
            nn.init.trunc_normal_(self.position, std=0.02)
            self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        qs = []
        for k in range(queries):
            g = torch.Generator().manual_seed(base+310+k)
            qs.append(torch.randn(dim, generator=g)*0.02)
        self.query = nn.Parameter(torch.stack(qs))

    def forward(self, patch, return_attention=False):
        assert patch.shape[1:] == (PATCHES, self.query.shape[1])
        values = self.norm(patch)
        keys = values+self.position.to(values.dtype)
        q = self.query.to(values.dtype).unsqueeze(0).expand(len(patch), -1, -1)
        pooled, weights = self.attention(q, keys, values, need_weights=return_attention, average_attn_weights=True)
        return (pooled, weights) if return_attention else pooled


class AuxHeads(nn.Module):
    def __init__(self, group, dim=1024, seed=17):
        super().__init__()
        cfg = GROUPS[group]
        self.group, self.cfg = group, cfg
        base = int(seed)*10000
        # All initialization runs in a separate RNG stream and consumes no global random numbers of the main model or data.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(base+300)
            if cfg['pool']:
                self.pool = PatchPool(1+(len(TARGETS) if cfg['oarsi'] == 'pool' else 0), dim, seed=seed)
                self.kl_pool = nn.Linear(dim, 5, bias=False)
                nn.init.zeros_(self.kl_pool.weight)  # At step 0 the KL output is bit-identical to the CLS-only head.
            if cfg['oarsi'] is not None:
                torch.manual_seed(base+320)
                self.oarsi_weight = nn.Parameter(torch.empty(len(TARGETS), 4, dim))
                self.oarsi_bias = nn.Parameter(torch.zeros(len(TARGETS), 4))
                bound = 1/math.sqrt(dim)
                nn.init.uniform_(self.oarsi_weight, -bound, bound)

    def forward(self, cls, patch, return_attention=False):
        out = {'kl_extra': None, 'oarsi': None, 'attention': None}
        pooled = None
        if self.cfg['pool']:
            r = self.pool(patch, return_attention)
            pooled, out['attention'] = r if return_attention else (r, None)
            out['kl_extra'] = self.kl_pool(pooled[:, 0])
        if self.cfg['oarsi'] == 'cls':
            out['oarsi'] = torch.einsum('bd,tcd->btc', cls, self.oarsi_weight.to(cls.dtype))+self.oarsi_bias.to(cls.dtype)
        elif self.cfg['oarsi'] == 'pool':
            out['oarsi'] = torch.einsum('btd,tcd->btc', pooled[:, 1:], self.oarsi_weight.to(cls.dtype))+self.oarsi_bias.to(cls.dtype)
        return out


def extra_parameters(group, dim=1024, heads=8):
    cfg = GROUPS[group]
    n = 0
    if cfg['pool']:
        k = 1+(len(TARGETS) if cfg['oarsi'] == 'pool' else 0)
        n += 2*dim + PATCHES*dim + (3*dim*dim+3*dim) + (dim*dim+dim) + k*dim + 5*dim
    if cfg['oarsi'] is not None:
        n += len(TARGETS)*4*dim + len(TARGETS)*4
    return n


def qwk(y, pred, classes):
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    if len(y) == 0:
        return None
    o = np.zeros((classes, classes))
    np.add.at(o, (y, pred), 1)
    w = np.subtract.outer(np.arange(classes), np.arange(classes))**2/(classes-1)**2
    e = np.outer(o.sum(1), o.sum(0))/o.sum()
    denom = (w*e).sum()
    return None if denom == 0 else float(1-(w*o).sum()/denom)


def auc(scores, positive):
    s, p = np.asarray(scores, float), np.asarray(positive, bool)
    n1, n0 = int(p.sum()), int((~p).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j+1 < len(s) and sorted_s[j+1] == sorted_s[i]:
            j += 1
        ranks[order[i:j+1]] = (i+j)/2+1
        i = j+1
    return float((ranks[p].sum()-n1*(n1+1)/2)/(n1*n0))


def ece(probabilities, labels, bins=15):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels)
    confidence, correct = p.max(1), (p.argmax(1) == y)
    indexes = np.minimum((confidence*bins).astype(int), bins-1)
    return float(sum(abs(confidence[indexes == b].mean()-correct[indexes == b].mean())*(indexes == b).mean()
                     for b in range(bins) if np.any(indexes == b)))


def oarsi_metrics(probabilities, targets):
    pred = np.asarray(probabilities).argmax(-1)
    t = np.asarray(targets)
    result = {}
    for j, name in enumerate(TARGETS):
        m = t[:, j] >= 0
        result[name] = {'n': int(m.sum()), 'qwk': qwk(t[m, j], pred[m, j], 4),
                        'accuracy': None if not m.any() else float((t[m, j] == pred[m, j]).mean())}
    return result


def flip_consistency(prob_original, prob_flipped):
    a = np.asarray(prob_original).argmax(-1)
    b = np.asarray(prob_flipped).argmax(-1)
    return {'same': float((a == b).mean()), 'swapped': float((a[:, SWAP] == b).mean())}


def chunked_vjp(tail, inputs, cotangents, params, chunk):
    totals = [[torch.zeros_like(p) for p in params] for _ in cotangents]
    for start in range(0, len(inputs), chunk):
        out = tail(inputs[start:start+chunk])
        for k, cot in enumerate(cotangents):
            grads = torch.autograd.grad(out, params, grad_outputs=cot[start:start+chunk].to(out),
                                        retain_graph=k < len(cotangents)-1, allow_unused=True)
            for acc, g in zip(totals[k], grads):
                if g is not None:
                    acc.add_(g)
    return totals


def global_norm(tensors):
    return float(torch.sqrt(sum((t.double()**2).sum() for t in tensors)))
