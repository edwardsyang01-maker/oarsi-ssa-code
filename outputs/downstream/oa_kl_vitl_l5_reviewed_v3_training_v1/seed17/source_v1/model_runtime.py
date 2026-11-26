import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import load_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTRopePositionEmbedding
import deepspeed
from batch_policy import lr_factor
ROOT=Path('/path/to/drai')
HERE=Path(__file__).resolve().parent
PLAN=json.loads((HERE/'plan.json').read_text())
WEIGHTS=ROOT/PLAN['weights_directory']
LOCAL_RANK=int(os.environ.get('LOCAL_RANK','0'))

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)

def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def digest(obj):
    h = hashlib.sha256()
    def add(x):
        if isinstance(x, torch.Tensor):
            t = x.detach().contiguous().cpu()
            h.update(str((tuple(t.shape), str(t.dtype))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=lambda v: str(v)):
                h.update(str(k).encode()); add(x[k])
        elif isinstance(x, (list, tuple)):
            for item in x: add(item)
        else:
            h.update(repr(x).encode())
    add(obj)
    return h.hexdigest()

class ViTLKL(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(PLAN['seed'])
            self.head = nn.Linear(PLAN['hidden_size'], PLAN['head_outputs'])
        self.trainable_blocks = int(PLAN['trainable_blocks'])
        assert 1 <= self.trainable_blocks <= len(encoder.layer)
        processor = json.loads((WEIGHTS / 'preprocessor_config.json').read_text())
        # Plain tuples keep normalization constants in FP32 even when DS halves model buffers.
        self.mean_values = tuple(processor['image_mean'])
        self.std_values = tuple(processor['image_std'])
        encoder.requires_grad_(False)
        for layer in encoder.layer[-self.trainable_blocks:]: layer.requires_grad_(True)
        encoder.norm.requires_grad_(True)
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        for layer in encoder.layer[:-self.trainable_blocks]: layer.gradient_checkpointing = False
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        if mode:
            for layer in self.encoder.layer[-self.trainable_blocks:]: layer.train(True)
            self.encoder.norm.train(True)
        self.head.train(mode)
        return self

    def forward(self, images):
        assert tuple(images.shape[1:]) == (1, 448, 448)
        mean = torch.tensor(self.mean_values, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std_values, device=images.device).view(1, 3, 1, 1)
        x = ((images.float().expand(-1, 3, -1, -1) - mean) / std).to(self.head.weight.dtype)
        hidden = self.encoder(pixel_values=x, return_dict=True).last_hidden_state
        assert tuple(hidden.shape) == (len(images), 789, PLAN['hidden_size'])
        return self.head(hidden[:, 0])

def create_model():
    cfg = DINOv3ViTConfig.from_dict(json.loads((WEIGHTS / 'config.json').read_text()))
    cfg._attn_implementation = 'sdpa'
    with torch.device('meta'):
        enc = DINOv3ViTModel(cfg)
    state = load_file(str(WEIGHTS / 'model.safetensors'), device='cpu')
    enc.load_state_dict(state, strict=True, assign=True)
    del state
    enc.rope_embeddings = DINOv3ViTRopePositionEmbedding(cfg).eval()
    model = ViTLKL(enc)
    assert all(not p.is_meta for p in list(model.parameters()) + list(model.buffers()))
    assert sum(p.numel() for p in model.parameters()) == PLAN['expected_total_parameters']
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == PLAN['expected_trainable_parameters']
    return model

def engine_init():
    model = create_model()
    groups = []
    if PLAN['layer_lr_decay'] == 1.0:
        groups.append({'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': PLAN['encoder_lr']})
    else:
        layers = list(model.encoder.layer)
        for index in range(len(layers) - PLAN['trainable_blocks'], len(layers)):
            groups.append({'params': list(layers[index].parameters()),
                           'lr': PLAN['encoder_lr'] * PLAN['layer_lr_decay'] ** (len(layers)-1-index)})
        groups.append({'params': list(model.encoder.norm.parameters()), 'lr': PLAN['encoder_lr']})
    groups.append({'params': list(model.head.parameters()), 'lr': PLAN['head_lr']})
    grouped = [p for group in groups for p in group['params']]
    assert len({id(p) for p in grouped}) == len(grouped)
    assert {id(p) for p in grouped} == {id(p) for p in model.parameters() if p.requires_grad}
    optim = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=PLAN['weight_decay'], foreach=False)
    counter = {'calls': 0}
    def applied(_opt, _args, _kwargs): counter['calls'] += 1
    optim.register_step_post_hook(applied)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda step: lr_factor(step, PLAN['steps_per_epoch'], PLAN['epochs'], PLAN['warmup_epochs'], PLAN['min_lr_ratio']))
    args = argparse.Namespace(local_rank=LOCAL_RANK)
    engine, _, _, _ = deepspeed.initialize(args=args, model=model, optimizer=optim,
                                          config=str(HERE / 'deepspeed_config.json'), lr_scheduler=scheduler)
    assert engine.module.head.weight.dtype == torch.bfloat16 and float(engine.optimizer.loss_scale) == 1.0
    assert engine.zero_optimization_stage() == 3 and engine.dp_world_size == 2
    assert engine.train_micro_batch_size_per_gpu() == 32 and engine.gradient_accumulation_steps() == 1
    assert not engine.optimizer.offload_optimizer and not engine.optimizer.offload_param
    assert all(hasattr(p, 'ds_tensor') for p in engine.module.parameters())
    return engine, counter

def fingerprints(engine):
    inner = engine.optimizer.optimizer
    masters = engine.optimizer.fp32_partitioned_groups_flat
    # ZeRO temporarily clears optimizer param_groups between steps, so the
    # ordinary torch state_dict mapping cannot be assumed to be populated.
    adam = {'state': [inner.state.get(p, {}) for p in masters],
            'groups': [{k: v for k, v in g.items() if k != 'params'} for g in inner.param_groups]}
    return {
        'model_local_shards': digest({n: p.ds_tensor for n, p in engine.module.named_parameters()}),
        'model_buffers': digest(dict(engine.module.named_buffers())),
        'fp32_master_shards': digest(engine.optimizer.fp32_partitioned_groups_flat),
        'adam_state': digest(adam),
        'loss_scale': float(engine.optimizer.loss_scale)
    }
