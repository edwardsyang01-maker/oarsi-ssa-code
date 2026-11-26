import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE/'spec.json').read_text())
ROOT = Path(SPEC['project'])
BASE = ROOT/SPEC['base_source']
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT))
import deepspeed
import model_runtime as original
from model_runtime import file_sha, write_json, digest, fingerprints
from batch_policy import rank_batches, lr_factor
from objective_policy import training_terms, training_augmentation
from storage import Checkpoints, torch_save_atomic
from sampler import BoardSampler
from downstream.oa_kl_unified_v1.training_core import classification_metrics, _order
from core import (GROUPS, TARGETS, TOKENS, REGISTERS, AuxHeads, extra_parameters, canonical_flip,
                  random_flip)

PLAN = copy.deepcopy(original.PLAN)
PLAN.update({k: SPEC[k] for k in ('seed', 'epochs', 'steps_per_epoch', 'total_steps', 'counts')})
original.PLAN = PLAN
# The random horizontal flip is drawn here first (labels must be swapped for flipped samples);
# the other augmentations reuse the original function with its own flip turned off.
FLIP_PROBABILITY = float(PLAN['augmentation_parameters']['horizontal_flip_probability'])
AUG_PLAN = copy.deepcopy(PLAN)
AUG_PLAN['augmentation_parameters']['horizontal_flip_probability'] = 0.0
RANK = int(os.environ.get('RANK', 0))
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
DEVICE = torch.device('cuda', LOCAL_RANK)
GROUP = None


class Model(original.ViTLKL):
    def __init__(self, encoder):
        super().__init__(encoder)
        self.group = GROUP
        if GROUPS[GROUP]['pool'] or GROUPS[GROUP]['oarsi'] is not None:
            self.aux = AuxHeads(GROUP)
        else:
            self.aux = None

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'aux', None) is not None:
            self.aux.train(mode)
        return self

    def forward(self, images, return_parts=False, return_attention=False, return_hidden=False):
        assert tuple(images.shape[1:]) == (1, 448, 448)
        mean = torch.tensor(self.mean_values, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std_values, device=images.device).view(1, 3, 1, 1)
        x = ((images.float().expand(-1, 3, -1, -1)-mean)/std).to(self.head.weight.dtype)
        hidden = self.encoder(pixel_values=x, return_dict=True).last_hidden_state
        assert tuple(hidden.shape) == (len(images), TOKENS, 1024)
        cls, patch = hidden[:, 0], hidden[:, 1+REGISTERS:]
        logits = self.head(cls)
        parts = {'kl_extra': None, 'oarsi': None, 'attention': None}
        if self.aux is not None:
            parts = self.aux(cls, patch, return_attention)
            if parts['kl_extra'] is not None:
                logits = logits+parts['kl_extra']
        if return_hidden:
            return logits, parts, hidden
        if return_parts or return_attention:
            return logits, parts
        return logits


def setup():
    assert Path.cwd() == ROOT and int(os.environ['WORLD_SIZE']) == 2
    for name, expected in PLAN['expected_versions'].items():
        assert importlib.metadata.version(name) == expected
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.cuda.set_device(LOCAL_RANK)
    random.seed(SPEC['seed']); np.random.seed(SPEC['seed']); torch.manual_seed(SPEC['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_bf16_supported()
    assert 'GPU-'+str(torch.cuda.get_device_properties(LOCAL_RANK).uuid).removeprefix('GPU-') == PLAN['gpu_uuids'][RANK]
    deepspeed.init_distributed(dist_backend='nccl')


def engine_init(group):
    global GROUP
    GROUP = group
    original.ViTLKL = Model
    extra = extra_parameters(group)
    PLAN['expected_total_parameters'] += extra
    PLAN['expected_trainable_parameters'] += extra
    model = original.create_model()
    actual_extra = 0 if model.aux is None else sum(p.numel() for p in model.aux.parameters())
    assert actual_extra == extra, 'EXTRA_PARAMETER_COUNT_MISMATCH'
    groups = []
    layers = list(model.encoder.layer)
    for i in range(len(layers)-6, len(layers)):
        groups.append({'params': list(layers[i].parameters()), 'lr': PLAN['encoder_lr']*PLAN['layer_lr_decay']**(len(layers)-1-i)})
    groups += [{'params': list(model.encoder.norm.parameters()), 'lr': PLAN['encoder_lr']},
               {'params': list(model.head.parameters()), 'lr': PLAN['head_lr']}]
    if model.aux is not None:
        groups.append({'params': list(model.aux.parameters()), 'lr': SPEC['new_module_lr']})
    assert {id(p) for g in groups for p in g['params']} == {id(p) for p in model.parameters() if p.requires_grad}
    opt = torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8, weight_decay=.05, foreach=False)
    counter = {'calls': 0}
    def applied(*_):
        counter['calls'] += 1
    opt.register_step_post_hook(applied)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: lr_factor(step, SPEC['steps_per_epoch'], 20, 1, .01))
    engine, _, _, _ = deepspeed.initialize(args=argparse.Namespace(local_rank=LOCAL_RANK), model=model,
        optimizer=opt, config=str(BASE/'deepspeed_config.json'), lr_scheduler=scheduler)
    assert engine.zero_optimization_stage() == 3 and engine.dp_world_size == 2
    assert engine.train_micro_batch_size_per_gpu() == 32 and engine.gradient_accumulation_steps() == 1
    assert not engine.optimizer.offload_optimizer and not engine.optimizer.offload_param
    assert engine.module.head.weight.dtype == torch.bfloat16
    torch.cuda.manual_seed(SPEC['seed']+1009*RANK)
    return engine, counter


def augment(x, targets, swap):
    mask = torch.rand(len(x), device=x.device) < FLIP_PROBABILITY
    flipped, t = random_flip(x, targets, mask, swap)
    return training_augmentation(flipped, AUG_PLAN), t, mask


def rng_capture():
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy_name': n[0], 'numpy_keys': torch.tensor(n[1].astype(np.int64)),
            'numpy_tail': n[2:], 'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(LOCAL_RANK)}


def rng_restore(state):
    random.setstate(state['python'])
    np.random.set_state((state['numpy_name'], state['numpy_keys'].numpy().astype(np.uint32), *state['numpy_tail']))
    torch.set_rng_state(state['cpu']); torch.cuda.set_rng_state(state['cuda'], LOCAL_RANK)


def finite(values, code):
    ok = torch.ones((), device=DEVICE, dtype=torch.int32)
    for value in values:
        ok *= torch.isfinite(value).all().to(torch.int32)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    assert bool(ok), code


class Dataset:
    def __init__(self, split):
        assert split in ('train', 'val'), 'TEST_ACCESS_FORBIDDEN'
        path = ROOT/SPEC[split+'_manifest']
        assert file_sha(path) == SPEC[split+'_sha256'], 'MANIFEST_IDENTITY_CHANGED'
        self.rows = [json.loads(line) for line in path.open()]
        assert len(self.rows) == SPEC['counts'][split]['knees']
        assert len({r['key'] for r in self.rows}) == len(self.rows)
        assert all(r['split'] == split and r['engineering_input_ready'] for r in self.rows)
        assert all(r['source_is_native_pixels'] and not r['resized_from_224'] and r['shape'] == [1, 448, 448] for r in self.rows)
        assert all(r['detector_expansion'] == SPEC['detector_expansion'] and r['eligibility'] == 'usable_technical_qa' for r in self.rows)
        assert all(r['cleanup_status'] in ('accepted_cleaned', 'accepted_unchanged') for r in self.rows), 'HELD_IMAGE_IN_USABLE_DATASET'
        assert all(r['unmasked_pixels_bit_exact'] for r in self.rows)
        assert all(r['clinical_side'] in ('R', 'L') for r in self.rows)
        self.labels = np.asarray([int(r['kl']) for r in self.rows], dtype=np.int64)
        assert np.bincount(self.labels, minlength=5).tolist() == SPEC[split+'_class_counts']
        patients = sorted({r['patient_id'] for r in self.rows})
        assert len(patients) == SPEC['counts'][split]['patients']
        self.patient_names = set(patients)
        lookup = {p: i for i, p in enumerate(patients)}
        self.patient = np.asarray([lookup[r['patient_id']] for r in self.rows], dtype=np.int64)
        self.is_left = np.asarray([r['clinical_side'] == 'L' for r in self.rows])
        from core import load_targets
        self.targets = load_targets(self.rows, ROOT/SPEC[split+'_oarsi'], SPEC[split+'_oarsi_sha256'])
        self.identity = digest({'manifest': SPEC[split+'_sha256'], 'oarsi': SPEC[split+'_oarsi_sha256']})

    def __len__(self):
        return len(self.rows)

    def batch(self, ids):
        xs = []
        for i in ids:
            path = Path(self.rows[int(i)]['gray01_npy'])
            path = path if path.is_absolute() else ROOT/path
            image = np.load(path, allow_pickle=False)
            assert image.shape == (1, 448, 448) and image.dtype == np.float32
            assert np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
            xs.append(torch.from_numpy(image))
        x = canonical_flip(torch.stack(xs).to(DEVICE), torch.as_tensor(self.is_left[ids]))
        return x, torch.as_tensor(self.labels[ids], device=DEVICE), torch.as_tensor(self.targets[ids], device=DEVICE)


def synthetic(ids):
    xs = [torch.rand(1, 448, 448, generator=torch.Generator().manual_seed(80000+int(i))) for i in ids]
    x = canonical_flip(torch.stack(xs).to(DEVICE), torch.as_tensor([int(i) % 2 == 1 for i in ids]))
    t = torch.as_tensor([[((int(i)+j) % 5)-1 for j in range(len(TARGETS))] for i in ids], device=DEVICE)
    return x, torch.as_tensor([int(i) % 5 for i in ids], device=DEVICE), t


def binding(group):
    return {'group': group, 'spec_sha256': file_sha(HERE/'spec.json'),
            'source': {p.name: file_sha(p) for p in sorted(HERE.glob('*.py'))},
            'base_source': {p.name: file_sha(p) for p in sorted(BASE.glob('*.py'))},
            'deepspeed_sha256': file_sha(BASE/'deepspeed_config.json'),
            'frozen_plan_sha256': file_sha(HERE/SPEC['frozen_plan']),
            'frozen_plan_amendment_sha256': file_sha(HERE/SPEC['frozen_plan_amendment'])}
