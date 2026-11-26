import argparse
import json
import math
import time
import traceback
import numpy as np
import torch
import torch.distributed as dist
from runtime import (SPEC, PLAN, ROOT, HERE, RANK, DEVICE, setup, engine_init, Dataset, synthetic, finite,
                     rng_capture, rng_restore, digest, fingerprints, file_sha, write_json, Checkpoints,
                     BoardSampler, rank_batches, training_terms, classification_metrics, _order, binding,
                     augment, deepspeed)
from core import (GROUPS, TARGETS, REGISTERS, AuxHeads, lambda_at, oarsi_loss, auc, ece, oarsi_metrics,
                  flip_consistency, chunked_vjp, global_norm)


def frozen_lambda(group):
    if GROUPS[group]['oarsi'] is None:
        return 0.0
    root = ROOT/SPEC['output']
    lock = json.loads((root/SPEC['lambda_lock']).read_text())
    assert lock['spec_sha256'] == file_sha(HERE/'spec.json') and lock['train_sha256'] == SPEC['train_sha256']
    source = SPEC['lambda_calibration_groups'][group]
    for r in (0, 1):
        path = root/'calibration'/f'calibration_{source}_rank{r}.json'
        assert file_sha(path) == lock['calibration_sha256'][source][str(r)], 'CALIBRATION_EVIDENCE_CHANGED'
    value = lock['lambda'][source]
    assert math.isfinite(value) and value > 0
    return value


def initial_state(group, data_identity, lam):
    return {'epoch': 1, 'next_batch': 0, 'step': 0, 'samples_seen': 0, 'group': group, 'lambda': lam,
            'data_identity': data_identity, 'history': [], 'best_qwk': None, 'best_epoch': None,
            'best_tag': None, 'epoch_loss_sum': 0., 'epoch_aux_sum': 0., 'epoch_samples': 0,
            'epoch_flips': 0}


def signatures(engine, state):
    return {**fingerprints(engine), 'scheduler': digest(engine.lr_scheduler.state_dict()),
            'amp': digest(vars(engine.optimizer.loss_scaler)), 'rng': digest(rng_capture()),
            'state': digest(state), 'engine_steps': engine.global_steps}


def plain_progress(state):
    return {k: state[k] for k in ('epoch', 'next_batch', 'step', 'samples_seen', 'best_tag')}


def save(engine, manager, state, tag):
    hashes = [None, None]; dist.all_gather_object(hashes, digest(state))
    assert hashes[0] == hashes[1], 'STATE_RANK_DIVERGENCE'
    progress = plain_progress(state)
    local = {'progress': progress, 'state': state, 'rng': rng_capture(), 'evidence': signatures(engine, state)}
    return manager.save(engine, progress, local, tag)


def restore(engine, manager, data_identity):
    progress, local, receipt = manager.load(engine)
    state = local['state']; rng_restore(local['rng'])
    assert state['data_identity'] == data_identity, 'DATA_IDENTITY_CHANGED'
    assert state['lambda'] == frozen_lambda(state['group']), 'LAMBDA_LOCK_CHANGED_ON_RESUME'
    actual = signatures(engine, state)
    assert actual == local['evidence'], 'RESTORED_COMPONENT_MISMATCH:'+str([k for k in actual if actual[k] != local['evidence'][k]])
    assert plain_progress(state) == progress
    return state, receipt


def update(engine, counter, state, x, y, t, global_count):
    engine.train()
    group = state['group']
    image, train_targets, flipped = augment(x, t, GROUPS[group]['swap'])
    logits, parts = engine(image, return_parts=True)
    terms = training_terms(logits, y, PLAN, global_count)
    if parts['oarsi'] is not None:
        aux = oarsi_loss(parts['oarsi'], train_targets)
    else:
        aux = torch.zeros(len(y), device=DEVICE)
    weight = lambda_at(state['lambda'], state['epoch'])
    per_sample = terms['objective']+weight*aux
    finite([logits, per_sample], 'NONFINITE_LOSS')
    loss = per_sample.sum()*dist.get_world_size()/global_count
    before = counter['calls']
    engine.backward(loss); engine.step()
    applied = torch.tensor(int(counter['calls'] > before), device=DEVICE)
    dist.all_reduce(applied)
    assert int(applied) == 2, 'OPTIMIZER_UPDATE_MISMATCH'
    finite([t_ for g in engine.optimizer.get_groups_without_padding(engine.optimizer.fp32_partitioned_groups_flat) for t_ in g], 'NONFINITE_MASTER')
    stats = torch.tensor([per_sample.detach().sum(), aux.detach().sum(), len(y), flipped.sum()], dtype=torch.float64, device=DEVICE)
    dist.all_reduce(stats)
    assert int(stats[2]) == global_count
    state['step'] += 1; state['next_batch'] += 1
    state['samples_seen'] += global_count; state['epoch_samples'] += global_count
    state['epoch_loss_sum'] += float(stats[0]); state['epoch_aux_sum'] += float(stats[1])
    state['epoch_flips'] += int(stats[3])
    return {'objective_mean': float(stats[0])/global_count, 'auxiliary_mean': float(stats[1])/global_count,
            'effective_lambda': weight}


def evaluate(engine, dataset, subset=None, extra_flip=False, attention=False):
    saved, mode = rng_capture(), engine.training
    engine.eval()
    ids = list(range(len(dataset))) if subset is None else list(map(int, subset))
    keys = ['ids', 'labels', 'probabilities', 'oarsi_probabilities', 'attention']
    local = {k: [] for k in keys}
    with torch.no_grad():
        for batch in rank_batches(ids, rank=RANK, pad_empty=True):
            x, y, _ = dataset.batch(batch['indices'])
            if extra_flip:
                x = x.flip(-1)
            logits, parts = engine(x, return_parts=True, return_attention=attention)
            finite([logits], 'NONFINITE_VALIDATION')
            n = batch['valid_count']
            local['ids'].extend(batch['indices'][:n]); local['labels'].extend(y.cpu().tolist()[:n])
            local['probabilities'].extend(logits.float().softmax(1).cpu().tolist()[:n])
            if parts['oarsi'] is not None:
                local['oarsi_probabilities'].extend(parts['oarsi'].float().softmax(-1).cpu().tolist()[:n])
            if attention and parts['attention'] is not None:
                local['attention'].extend(parts['attention'].float().cpu().tolist()[:n])
    rows = [None, None]; dist.all_gather_object(rows, local)
    merged = {k: sum([a[k] for a in rows], []) for k in keys}
    order = np.argsort(merged['ids'])
    result = {k: np.asarray(v)[order] for k, v in merged.items() if len(v)}
    assert result['ids'].tolist() == sorted(ids), 'EVAL_DUPLICATE_OR_MISSING'
    engine.train(mode); rng_restore(saved)
    return result


def metrics_for(rows, targets=None):
    p, y = rows['probabilities'], rows['labels'].astype(int)
    ce = -np.log(np.maximum(p[np.arange(len(y)), y], np.finfo(float).tiny))
    result = classification_metrics(y, p, ce)
    error = np.abs(p.argmax(1)-y)
    result.update(ece=ece(p, y), cross_two_error_rate=float((error >= 2).mean()),
                  kl2_as_kl0_count=int(((y == 2) & (p.argmax(1) == 0)).sum()),
                  auc_kl_ge2=auc(p[:, 2:].sum(1), y >= 2))
    if 'oarsi_probabilities' in rows and targets is not None:
        result['oarsi'] = oarsi_metrics(rows['oarsi_probabilities'], targets[rows['ids']])
    return result


def attention_panel(dataset):
    return sorted(np.random.default_rng(SPEC['attention_panel']['seed']).choice(len(dataset), SPEC['attention_panel']['images'], replace=False).tolist())


def _float_kwargs(kwargs):
    def conv(v):
        if torch.is_tensor(v):
            return v.detach().float()
        if isinstance(v, (tuple, list)):
            return type(v)(conv(u) for u in v)
        return v
    return {k: conv(v) for k, v in kwargs.items()}


def gradient_probe(engine, train, group):
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTLayer
    saved_rng, mode = rng_capture(), engine.training
    engine.eval()
    selected = np.random.default_rng(170204).choice(len(train), SPEC['calibration_images'], replace=False).tolist()
    encoder = engine.module.encoder
    blocks = list(encoder.layer[-PLAN['trainable_blocks']:])
    expected = sum(getattr(p, 'ds_numel', p.numel()) for p in encoder.parameters() if p.requires_grad)
    modules = [engine.module.head, engine.module.aux]
    shared_ds = [p for m in blocks+[encoder.norm] for p in m.parameters()]
    with deepspeed.zero.GatheredParameters([p for m in modules for p in m.parameters()]+shared_ds, modifier_rank=None):
        head = torch.nn.Linear(1024, 5).float()
        head.load_state_dict({k: v.detach().float().cpu() for k, v in modules[0].state_dict().items()})
        aux = AuxHeads(group).float()
        aux.load_state_dict({k: v.detach().float().cpu() for k, v in modules[1].state_dict().items()})
        copies = torch.nn.ModuleList([DINOv3ViTLayer(encoder.config) for _ in blocks])
        for new, old in zip(copies, blocks):
            new.load_state_dict({k: v.detach().float().cpu() for k, v in old.state_dict().items()}, strict=True)
        norm = torch.nn.LayerNorm(encoder.config.hidden_size, eps=encoder.config.layer_norm_eps)
        norm.load_state_dict({k: v.detach().float().cpu() for k, v in encoder.norm.state_dict().items()}, strict=True)
    aux.eval(); head.eval()
    copies = copies.float().to(DEVICE).eval(); norm = norm.float().to(DEVICE).eval()
    shared = [p for m in list(copies)+[norm] for p in m.parameters()]
    for p in shared:
        p.requires_grad_(True)
    count = sum(p.numel() for p in shared)
    assert count == expected, ('SHARED_PARAMETER_COUNT', count, expected)
    captured = {}

    def grab(module, args, kwargs):
        captured['h'] = (args[0] if args else kwargs['hidden_states']).detach()
        captured['kw'] = {k: v for k, v in kwargs.items() if k != 'hidden_states'}
        captured['extra_args'] = len(args)-1 if args else 0

    handle = blocks[0].register_forward_pre_hook(grab, with_kwargs=True)
    chunk = SPEC['calibration_microbatch']
    ratios, norms = [], []
    try:
        for b in rank_batches(selected, rank=RANK):
            x, y, t = train.batch(b['indices'])
            captured.clear()
            with torch.no_grad():
                _, _, hidden = engine(x, return_hidden=True)
            assert captured.get('extra_args') == 0, 'UNEXPECTED_POSITIONAL_LAYER_ARGS'
            kw = _float_kwargs(captured['kw'])
            h = captured['h'].float()
            n = len(h)

            def tail(z):
                for block in copies:
                    z = block(z, **kw)
                return norm(z)

            with torch.no_grad():
                recomputed = torch.cat([tail(h[s:s+chunk]) for s in range(0, n, chunk)])
            ref = hidden.float()
            rel = float(torch.linalg.vector_norm(recomputed-ref)/torch.linalg.vector_norm(ref))
            assert rel < 0.05, ('FP32_RECOMPUTE_MISMATCH', rel)
            tokens = recomputed.cpu().requires_grad_(True)
            cls, patch = tokens[:, 0], tokens[:, 1+REGISTERS:]
            parts = aux(cls, patch)
            logits = head(cls)+(parts['kl_extra'] if parts['kl_extra'] is not None else 0)
            main = training_terms(logits, y.cpu(), PLAN, len(y))['objective'].mean()
            side = oarsi_loss(parts['oarsi'], t.cpu()).mean()
            dm = torch.autograd.grad(main, tokens, retain_graph=True)[0]
            ds = torch.autograd.grad(side, tokens)[0]
            token_main, token_side = float(torch.linalg.vector_norm(dm)), float(torch.linalg.vector_norm(ds))
            grads = chunked_vjp(tail, h, [dm.to(DEVICE), ds.to(DEVICE)], shared, chunk)
            gc, go = global_norm(grads[0]), global_norm(grads[1])
            del grads
            norms.append({'main': gc, 'oarsi': go, 'token_main_v1_measure': token_main, 'token_oarsi_v1_measure': token_side,
                          'fp32_recompute_relative_error': rel})
            if math.isfinite(gc) and math.isfinite(go) and gc > 0 and go > 1e-12:
                ratios.append(gc/go)
    finally:
        handle.remove()
    del copies, norm, shared
    torch.cuda.empty_cache()
    engine.train(mode); rng_restore(saved_rng)
    return {'ratios': ratios, 'batch_gradient_norms': norms, 'shared_parameter_count': count}


def calibration(engine, counter, run, group):
    assert GROUPS[group]['oarsi'] is not None and SPEC['lambda_calibration_groups'][group] == group
    train = Dataset('train')
    saved_rng, before, before_steps = rng_capture(), fingerprints(engine), counter['calls']
    t0 = time.perf_counter()
    probe = gradient_probe(engine, train, group)
    gathered = [None, None]; dist.all_gather_object(gathered, probe['ratios'])
    ratios = gathered[0]+gathered[1]
    assert len(ratios) >= SPEC['calibration_min_valid_batches'], 'INSUFFICIENT_VALID_CALIBRATION_BATCHES'
    value = SPEC['lambda_target']*float(np.median(ratios))
    assert math.isfinite(value) and value > 0, 'INVALID_LAMBDA'
    rng_restore(saved_rng)
    assert fingerprints(engine) == before and counter['calls'] == before_steps == 0, 'CALIBRATION_MUTATED_MODEL'
    write_json(run/f'calibration_{group}_rank{RANK}.json', {
        'group': group, 'lambda': value, 'lambda_target': SPEC['lambda_target'], 'all_ratios_main_over_oarsi': ratios,
        'local_batch_gradient_norms': probe['batch_gradient_norms'], 'valid_batches': len(ratios),
        'median_oarsi_to_main_at_lambda': float(np.median([value/r for r in ratios])),
        'gradient_location': SPEC['lambda_calibration_location'], 'shared_parameter_count': probe['shared_parameter_count'],
        'token_level_v1_ratios_main_over_oarsi': [r['token_main_v1_measure']/r['token_oarsi_v1_measure']
                                                  for r in probe['batch_gradient_norms'] if r['token_oarsi_v1_measure'] > 1e-12], 'random_augmentation': False,
        'training_images': SPEC['calibration_images'], 'real_optimizer_steps': 0, 'model_optimizer_rng_unchanged': True,
        'seconds': time.perf_counter()-t0, 'spec_sha256': file_sha(HERE/'spec.json'), 'binding': binding(group)})


def timed_update(engine, counter, state, ids, output, name):
    x, y, t = synthetic(ids)
    n = torch.tensor(len(ids), device=DEVICE); dist.all_reduce(n)
    start = time.perf_counter(); stats = update(engine, counter, state, x, y, t, int(n)); torch.cuda.synchronize()
    elapsed = torch.tensor(time.perf_counter()-start, device=DEVICE, dtype=torch.float64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    output.append({'name': name, 'local_n': len(ids), 'global_n': int(n), 'seconds': float(elapsed), **stats})


def preflight(engine, counter, run, group, mode, sampler):
    n = SPEC['counts']['train']['knees']
    tail = n % 64 or 64
    remaining = 6*64+tail
    identity = digest({'synthetic': n})
    manager = Checkpoints(run/'checkpoints', run/'recovery_copy', binding(group))
    timings = []
    if mode == 'preflight_save':
        state = initial_state(group, identity, frozen_lambda(group))
        start_batch = SPEC['steps_per_epoch']-7
        state.update(epoch=5, next_batch=start_batch, step=4*SPEC['steps_per_epoch']+start_batch,
                     samples_seen=4*n+n-remaining, epoch_samples=n-remaining)
        for start in range(n-remaining, n-(2*64+tail), 64):
            timed_update(engine, counter, state, list(range(start, start+64))[RANK::2], timings, 'full64_before_save')
        sampler.set_phase('checkpoint')
        t = time.perf_counter(); receipt = save(engine, manager, state, 's000001_prefinaltail')
        write_json(run/f'saved_rank{RANK}.json', {'receipt': receipt, 'checkpoint_seconds': time.perf_counter()-t,
                   'evidence': signatures(engine, state)})
    else:
        sampler.set_phase('restore')
        t = time.perf_counter(); state, _ = restore(engine, manager, identity)
        write_json(run/f'restored_rank{RANK}.json', {'full_state_exact': True, 'restore_seconds': time.perf_counter()-t,
                   'restored_lambda': state['lambda']})
    sampler.set_phase('continuation')
    snapshots = []
    for start in [n-(2*64+tail), n-(64+tail)]:
        timed_update(engine, counter, state, list(range(start, start+64))[RANK::2], timings, 'full64_after_save')
        snapshots.append(signatures(engine, state))
    timed_update(engine, counter, state, list(range(n-tail, n))[RANK::2], timings, f'tail{tail}')
    assert state['epoch_samples'] == n
    state.update(epoch=6, next_batch=0, epoch_samples=0, epoch_loss_sum=0., epoch_aux_sum=0., epoch_flips=0)
    snapshots.append(signatures(engine, state))
    timed_update(engine, counter, state, list(range(64))[RANK::2], timings, 'new_epoch_full64')
    timed_update(engine, counter, state, list(range(64, 69))[RANK::2], timings, 'uneven_tail5')
    # No-grad inference with attention output must not change any training state.
    before, calls = signatures(engine, state), counter['calls']
    engine.eval()
    with torch.no_grad():
        x, _, _ = synthetic(list(range(8))[RANK::2])
        logits, parts = engine(x, return_parts=True, return_attention=True)
        shapes = {'logits': list(logits.shape), 'oarsi': None if parts['oarsi'] is None else list(parts['oarsi'].shape),
                  'attention': None if parts['attention'] is None else list(parts['attention'].shape)}
    engine.train()
    assert signatures(engine, state) == before and counter['calls'] == calls, 'INFERENCE_MUTATED_STATE'
    snapshots.append(signatures(engine, state))
    expected_file = run/f'expected_continuation_rank{RANK}.json'
    if mode == 'preflight_save':
        write_json(expected_file, snapshots)
    else:
        expected = json.loads(expected_file.read_text())
        differences = [{k: {'expected': a[k], 'actual': b[k]} for k in a if a[k] != b[k]} for a, b in zip(expected, snapshots)]
        write_json(run/f'resume_comparison_rank{RANK}.json', {'exact': expected == snapshots, 'differences': differences})
        assert expected == snapshots, 'COLD_CONTINUATION_NOT_BITWISE_EXACT'
    write_json(run/f'{mode}_timing_rank{RANK}.json', {'rows': timings, 'synthetic_only': True,
        'actual_patient_optimizer_steps': 0, 'locked_lambda': state['lambda'], 'output_shapes': shapes,
        'swap_on_flip': GROUPS[group]['swap']})


def train_run(engine, counter, run, group, resume, sampler):
    train, val = Dataset('train'), Dataset('val')
    assert not train.patient_names & val.patient_names, 'PATIENT_OVERLAP'
    lam = frozen_lambda(group)
    manager = Checkpoints(run/'checkpoints', run/'recovery_copy', binding(group))
    state = initial_state(group, train.identity, lam)
    if resume:
        state, _ = restore(engine, manager, train.identity)
    panel = attention_panel(val)
    while state['epoch'] <= SPEC['epochs']:
        epoch = state['epoch']; order, _ = _order(len(train), SPEC['seed'], epoch-1)
        for b in rank_batches(order, rank=RANK, start_batch=state['next_batch']):
            sampler.set_phase('train')
            x, y, t = train.batch(b['indices'])
            update(engine, counter, state, x, y, t, b['global_count'])
            if state['step'] == 10 or state['step'] % 400 == 0:
                save(engine, manager, state, f"s{state['step']:06d}_e{epoch:02d}_b{state['next_batch']:03d}")
        assert state['epoch_samples'] == len(train) and state['next_batch'] == SPEC['steps_per_epoch']
        sampler.set_phase('validation')
        rows = evaluate(engine, val)
        metrics = metrics_for(rows, val.targets)
        extra = {}
        if GROUPS[group]['oarsi'] == 'pool':
            flipped = evaluate(engine, val, extra_flip=True)
            extra['flip_consistency'] = flip_consistency(rows['oarsi_probabilities'], flipped['oarsi_probabilities'])
            if RANK == 0:
                np.savez_compressed(run/f'val_extra_flip_epoch{epoch:02d}.npz', **flipped)
        if GROUPS[group]['pool']:
            maps = evaluate(engine, val, subset=panel, attention=True)
            if RANK == 0:
                np.savez_compressed(run/f'attention_panel_epoch{epoch:02d}.npz', **maps)
        row = {'epoch': epoch, 'step': state['step'], 'validation': metrics, **extra,
               'objective_mean': state['epoch_loss_sum']/len(train), 'oarsi_mean': state['epoch_aux_sum']/len(train),
               'effective_lambda': lambda_at(state['lambda'], epoch), 'random_flip_fraction': state['epoch_flips']/len(train)}
        state['history'].append(row)
        tag = f"s{state['step']:06d}_e{epoch+1:02d}_b000"
        if metrics['qwk'] is not None and (state['best_qwk'] is None or metrics['qwk'] > state['best_qwk']):
            state.update(best_qwk=metrics['qwk'], best_epoch=epoch, best_tag=tag)
        if RANK == 0:
            np.savez_compressed(run/f'val_epoch{epoch:02d}.npz', **rows)
            write_json(run/'epoch_history.json', state['history'])
        state.update(epoch=epoch+1, next_batch=0, epoch_loss_sum=0., epoch_aux_sum=0., epoch_samples=0, epoch_flips=0)
        sampler.set_phase('checkpoint'); save(engine, manager, state, tag)
    if RANK == 0:
        write_json(run/'completed.json', {'epochs': SPEC['epochs'], 'step': state['step'], 'best_epoch': state['best_epoch'],
            'best_checkpoint': state['best_tag'], 'test_accessed': False})
    return 'completed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=SPEC['groups'], required=True)
    parser.add_argument('--mode', choices=['preflight_save', 'preflight_resume', 'calibration', 'train'], required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.mode == 'train':
        auth_path = HERE/'training_authorization.json'
        assert auth_path.is_file(), 'FORMAL_TRAINING_NOT_AUTHORIZED'
        auth = json.loads(auth_path.read_text())
        assert auth.get('authorized') is True and auth['bindings'][args.group] == binding(args.group), 'TRAINING_AUTHORIZATION_MISMATCH'
    elif args.mode == 'calibration':
        assert args.group in ('P2', 'P3')
    name = f'calibration' if args.mode == 'calibration' else ('train_'+args.group if args.mode == 'train' else 'preflight_'+args.group)
    run = ROOT/SPEC['output']/name
    run.mkdir(parents=True, exist_ok=True)
    setup(); sampler = BoardSampler.for_uuid(PLAN['gpu_uuids'][RANK]).start()
    passed = False
    try:
        engine, counter = engine_init(args.group)
        sampler.set_phase(args.mode)
        if args.mode == 'calibration':
            calibration(engine, counter, run, args.group)
        elif args.mode == 'train':
            train_run(engine, counter, run, args.group, args.resume, sampler)
        else:
            preflight(engine, counter, run, args.group, args.mode, sampler)
        passed = True
        dist.barrier()
    except BaseException as exc:
        write_json(run/f'{args.mode}_{args.group}_failure_rank{RANK}.json', {'type': type(exc).__name__, 'error': str(exc)[:500]})
        traceback.print_exc()
    finally:
        write_json(run/f'{args.mode}_{args.group}_memory_rank{RANK}.json', {'passed': passed, 'board': sampler.stop(),
            'allocated_peak_bytes': torch.cuda.max_memory_allocated(), 'reserved_peak_bytes': torch.cuda.max_memory_reserved()})
        dist.destroy_process_group()
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
