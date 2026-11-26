import argparse
import datetime as dt
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
import numpy as np
import torch
import torch.distributed as dist
import deepspeed
from model_runtime import PLAN, ROOT, HERE, LOCAL_RANK, engine_init, create_model, digest, fingerprints, file_sha, write_json, utc
from batch_policy import rank_batches, loss_mean_multiplier
from storage import Checkpoints, torch_save_atomic
from sampler import BoardSampler
from loss_policy import class_weights
from objective_policy import training_terms, log_probabilities, training_augmentation

sys.path.insert(0, str(ROOT))
from downstream.oa_kl_unified_v1.data import NativeKneeDataset
from downstream.oa_kl_unified_v1.training_core import _order, classification_metrics

RANK = int(os.environ.get('RANK', 0))
DEVICE = torch.device('cuda', LOCAL_RANK)
CURRENT_BATCH = {}


def require_finite_all_ranks(tensors, code):
    local_ok = torch.ones((), dtype=torch.int32, device=DEVICE)
    for value in tensors:
        local_ok.mul_(torch.isfinite(value).all().to(torch.int32))
    dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
    if not bool(local_ok): raise FloatingPointError(code)



def verify_master_weights(engine):
    optimizer = engine.optimizer
    lean_groups = optimizer.get_groups_without_padding(optimizer.fp32_partitioned_groups_flat)
    tensors = [value for group in lean_groups for value in group]
    if not getattr(engine, '_padding_audit_written', False):
        write_json(engine._audit_run / f'master_partition_audit_rank{RANK}.json', {
            'at': utc(), 'rank': RANK,
            'all_elements': sum(v.numel() for v in optimizer.fp32_partitioned_groups_flat),
            'valid_elements': sum(v.numel() for v in tensors),
            'padding_elements': sum(sum(group) for group in optimizer.groups_padding),
            'nonfinite_with_padding': sum(int((~torch.isfinite(v)).sum()) for v in optimizer.fp32_partitioned_groups_flat),
            'nonfinite_valid_parameters': sum(int((~torch.isfinite(v)).sum()) for v in tensors)})
        engine._padding_audit_written = True
    require_finite_all_ranks(tensors, 'NONFINITE_MASTER_WEIGHTS')


def rng_capture():
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy_name': n[0],
            'numpy_keys': torch.tensor(n[1].astype(np.int64)), 'numpy_tail': n[2:],
            'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(LOCAL_RANK)}


def rng_restore(s):
    random.setstate(s['python'])
    np.random.set_state((s['numpy_name'], s['numpy_keys'].numpy().astype(np.uint32), *s['numpy_tail']))
    torch.set_rng_state(s['cpu']); torch.cuda.set_rng_state(s['cuda'], LOCAL_RANK)


def evidence(engine, progress):
    return {**fingerprints(engine), 'amp_full': digest(vars(engine.optimizer.loss_scaler)),
            'scheduler': digest(engine.lr_scheduler.state_dict()), 'cursor': digest(progress),
            'rng': digest(rng_capture())}


def initial_progress():
    return {'epoch': 0, 'next_batch': 0, 'step': 0, 'samples_seen': 0,
            'epoch_loss_sum': 0.0, 'epoch_ce_sum': 0.0, 'epoch_rps_sum': 0.0, 'epoch_correct': 0, 'epoch_samples': 0,
            'train_seconds': 0.0, 'validation_seconds': 0.0, 'checkpoint_seconds': 0.0,
            'skipped_amp_attempts': 0, 'best_key': None, 'best_epoch': None,
            'best_metrics': None, 'best_tag': None, 'history': []}


def synthetic_batch(indices):
    xs = [torch.rand(1, 448, 448, generator=torch.Generator().manual_seed(80000 + int(i))) for i in indices]
    return torch.stack(xs).to(DEVICE), torch.tensor([int(i) % 5 for i in indices], device=DEVICE)


def real_batch(dataset, indices):
    rows = [dataset[int(i)] for i in indices]
    x = torch.stack([torch.as_tensor(row['image']) for row in rows])
    y = torch.tensor([int(row['kl']) for row in rows], dtype=torch.long)
    assert x.shape == (len(indices), 1, 448, 448) and x.is_floating_point()
    assert torch.isfinite(x).all() and x.min() >= 0 and x.max() <= 1
    assert y.min() >= 0 and y.max() <= 4
    return x.to(DEVICE), y.to(DEVICE)


def train_step(engine, counter, x, y, global_count):
    engine.train()
    saved_rng = rng_capture()
    for attempt in range(PLAN['amp_max_attempts']):
        if attempt: rng_restore(saved_rng)
        before = counter['calls']
        images = training_augmentation(x, PLAN)
        output = engine(images)
        eligible = None
        if PLAN['head_type'] == 'corn_conditional':
            eligible = torch.clamp(y + 1, max=4).sum()
            dist.all_reduce(eligible)
        terms = training_terms(output, y, PLAN, global_count, None if eligible is None else int(eligible))
        losses = terms['objective']
        require_finite_all_ranks([output, losses], 'NONFINITE_TRAIN_LOSS')
        loss = losses.mean() * loss_mean_multiplier(len(y), global_count, 2)
        engine.backward(loss); engine.step()
        applied = counter['calls'] > before
        flag = torch.tensor(int(applied), device=DEVICE)
        dist.all_reduce(flag)
        assert int(flag) in (0, 2), 'RANK_UPDATE_DISAGREEMENT'
        if applied:
            verify_master_weights(engine)
            stats = torch.tensor([float(losses.detach().sum()), float((terms['predictions'] == y).sum()), len(y),
                                  float(terms['cross_entropy'].detach().sum()), float(terms['ordinal_rps'].detach().sum())],
                                 dtype=torch.float64, device=DEVICE)
            dist.all_reduce(stats)
            assert int(stats[2]) == global_count
            return {'loss_sum': float(stats[0]), 'correct': int(stats[1]), 'samples': int(stats[2]),
                    'ce_sum': float(stats[3]), 'rps_sum': float(stats[4]),
                    'skips': attempt, 'loss_scale': float(engine.optimizer.loss_scale)}
    raise RuntimeError('AMP_RETRY_LIMIT')


def collect_predictions(engine, dataset=None, synthetic_count=None):
    engine.eval()
    size = len(dataset) if dataset is not None else synthetic_count
    local = {'indices': [], 'labels': [], 'probabilities': [], 'ce': []}
    with torch.no_grad():
        for b in rank_batches(list(range(size)), rank=RANK, pad_empty=True):
            x, y = real_batch(dataset, b['indices']) if dataset is not None else synthetic_batch(b['indices'])
            logits = engine(x).float()
            require_finite_all_ranks([logits], 'NONFINITE_VALIDATION_LOGITS')
            logp = log_probabilities(logits, PLAN)
            p = logp.exp().cpu().numpy()
            ce = (-logp.gather(1, y[:, None]).squeeze(1)).cpu().numpy()
            n = b['valid_count']
            local['indices'].extend(b['indices'][:n]); local['labels'].extend(y.cpu().tolist()[:n])
            local['probabilities'].extend(p[:n].tolist()); local['ce'].extend(ce[:n].tolist())
    outputs = [None, None]
    dist.all_gather_object(outputs, local)
    merged = {k: sum([x[k] for x in outputs], []) for k in local}
    order = np.argsort(merged['indices'])
    arrays = {k: np.asarray(v)[order] for k, v in merged.items()}
    assert np.array_equal(arrays['indices'], np.arange(size)), 'VALIDATION_DUPLICATE_OR_MISSING'
    metrics = classification_metrics(arrays['labels'], arrays['probabilities'], arrays['ce'])
    if dataset is not None:
        for attribute in ['visit', 'clinical_side']:
            keys = [getattr(s, attribute) for s in dataset._specs]
            groups = {}
            for value in sorted(set(keys)):
                mask = np.asarray(keys) == value
                groups[value] = classification_metrics(arrays['labels'][mask], arrays['probabilities'][mask], arrays['ce'][mask])
            metrics['by_' + attribute] = groups
    return arrays, metrics


def save_best_delta(engine, path):
    state = {}
    for name, p in engine.module.named_parameters():
        if not p.requires_grad: continue
        with deepspeed.zero.GatheredParameters([p], modifier_rank=None):
            if RANK == 0: state[name] = p.detach().cpu().clone()
    if RANK == 0:
        state.update({k: v.detach().cpu().clone() for k, v in engine.module.named_buffers() if k in engine.module.state_dict()})
        torch_save_atomic({'format': 'ViTL_BF16_trainable_delta_v1', 'state': state,
                           'base_weights_sha256': PLAN['weights_sha256']['model.safetensors'],
                           'plan_sha256': file_sha(HERE / 'plan.json')}, path)
    dist.barrier()


def publish_best(progress, run, manager):
    if RANK == 0 and progress['best_epoch'] is not None:
        p = run / progress['best_delta_path']
        assert file_sha(p) == progress['best_delta_sha256']
        dest = manager.nas / 'artifacts' / p.name
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            temp = dest.with_suffix('.partial')
            shutil.copy2(p, temp)
            assert file_sha(temp) == progress['best_delta_sha256']
            os.replace(temp, dest)
        assert file_sha(dest) == progress['best_delta_sha256']
        write_json(run / 'best_selection.json', {'epoch': progress['best_epoch'], 'metrics': progress['best_metrics'],
            'checkpoint_tag': progress['best_tag'], 'delta_path': progress['best_delta_path'],
            'delta_sha256': progress['best_delta_sha256'], 'base_weights_sha256': PLAN['weights_sha256']['model.safetensors']})
    dist.barrier()


def export_best(progress, run, manager):
    result = None
    if RANK == 0:
        patch = torch.load(run / progress['best_delta_path'], map_location='cpu', weights_only=False)
        model = create_model().bfloat16().eval()
        missing, unexpected = model.load_state_dict(patch['state'], strict=False)
        frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
        assert not unexpected and set(missing) == frozen, 'BEST_DELTA_SCHEMA_MISMATCH'
        target = run / 'best_model.pt'
        torch_save_atomic({'format': 'drai_vitl_kl5_full_bf16_v1', 'model': model.state_dict(),
                           'plan': PLAN, 'epoch': progress['best_epoch'], 'validation': progress['best_metrics']}, target)
        dest = manager.nas / 'artifacts' / target.name
        temp = dest.with_suffix('.partial'); shutil.copy2(target, temp)
        h = file_sha(target); assert file_sha(temp) == h
        os.replace(temp, dest)
        result = {'path': str(target.relative_to(ROOT)), 'sha256': h,
                  'nas_path': str(dest), 'nas_sha256': file_sha(dest), 'bytes': target.stat().st_size}
        del model, patch
    box = [result]; dist.broadcast_object_list(box, src=0)
    return box[0]


def write_status(run, state, progress, sampler, extra=None):
    payload = {'state': state, 'updated_at': utc(), 'pid': os.getpid(),
               'start_ticks': int(Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19]),
               'rank': RANK, 'gpu_uuid': PLAN['gpu_uuids'][RANK], 'progress': progress,
               'cuda_allocated_peak_bytes': torch.cuda.max_memory_allocated(),
               'cuda_reserved_peak_bytes': torch.cuda.max_memory_reserved(), 'board': sampler.snapshot()}
    if extra: payload.update(extra)
    write_json(run / f'rank{RANK}_status.json', payload)


def save_transaction(engine, manager, progress, tag):
    local = {'rng': rng_capture(), 'progress': progress, 'evidence': evidence(engine, progress)}
    receipt = manager.save(engine, progress, local, tag)
    rng_restore(local['rng'])
    return receipt


def preflight(engine, counter, manager, run, mode):
    order, order_hash = _order(197, PLAN['seed'], 0)
    batches = rank_batches(order, rank=RANK)
    progress = initial_progress()
    if mode == 'preflight_save':
        for b in batches[:2]:
            x, y = synthetic_batch(b['indices']); train_step(engine, counter, x, y, b['global_count'])
            progress['step'] += 1; progress['next_batch'] += 1
        progress['order_sha256'] = order_hash
        checkpoint_start = time.perf_counter()
        cp = save_transaction(engine, manager, progress, 's000002_e00_b002')
        write_json(run / f'checkpoint_timing_rank{RANK}.json', {'seconds': time.perf_counter()-checkpoint_start, 'receipt': cp})
    else:
        progress, local, _ = manager.load(engine)
        rng_restore(local['rng'])
        actual = evidence(engine, progress)
        matches = {k: actual[k] == local['evidence'][k] for k in actual}
        write_json(run / f'restore_state_rank{RANK}.json', {'matches': matches})
        assert all(matches.values()), 'PREUPDATE_RESTORE_MISMATCH'
        assert progress['order_sha256'] == order_hash and progress['next_batch'] == 2
    update_results = []
    for b in rank_batches(order, rank=RANK, start_batch=progress['next_batch']):
        x, y = synthetic_batch(b['indices'])
        update_results.append(train_step(engine, counter, x, y, b['global_count']))
        progress['step'] += 1; progress['next_batch'] += 1
    final = {'evidence': evidence(engine, progress), 'updates': update_results}
    arrays, metrics = collect_predictions(engine, synthetic_count=17)
    final.update(prediction_sha=digest({k: torch.as_tensor(v) for k, v in arrays.items()}), metrics=metrics)
    if mode == 'preflight_save':
        write_json(run / f'golden_rank{RANK}.json', final)
    else:
        golden = json.loads((run / f'golden_rank{RANK}.json').read_text())
        checks = {'state_' + k: final['evidence'][k] == golden['evidence'][k] for k in final['evidence']}
        checks.update(updates_exact=final['updates'] == golden['updates'],
                      predictions_exact=final['prediction_sha'] == golden['prediction_sha'],
                      metrics_exact=metrics == golden['metrics'], validation_count=metrics['knees'] == 17)
        write_json(run / f'preflight_rank{RANK}.json', {'passed': all(checks.values()), 'checks': checks,
                                                     'actual': final, 'expected': golden})
        assert all(checks.values()), 'CONTINUOUS_RESUME_MISMATCH'
        measured = []
        for iteration in range(10):
            torch.cuda.synchronize(); begin = time.perf_counter()
            x, y = synthetic_batch(list(range(RANK*32, (RANK+1)*32)))
            train_step(engine, counter, x, y, 64)
            torch.cuda.synchronize()
            elapsed = torch.tensor(time.perf_counter()-begin, device=DEVICE, dtype=torch.float64)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            if iteration >= 2: measured.append(float(elapsed))
        torch.cuda.synchronize(); begin = time.perf_counter()
        collect_predictions(engine, synthetic_count=640)
        torch.cuda.synchronize()
        validation_seconds = time.perf_counter()-begin
        write_json(run / f'benchmark_rank{RANK}.json', {
            'synthetic_only': True, 'successful_measured_updates': len(measured),
            'global_batch': 64, 'seconds_per_update': measured,
            'mean_seconds_per_update': float(np.mean(measured)),
            'validation_samples': 640, 'validation_seconds': validation_seconds,
            'excludes_real_dataset_io': True})



def datasets():
    args = (ROOT, PLAN['manifest']['path'], PLAN['manifest']['sha256'], PLAN['cache_root'])
    train = NativeKneeDataset(*args, 'train', 448, allow_synthetic=False)
    val = NativeKneeDataset(*args, 'val', 448, allow_synthetic=False)
    patients = []
    for split, ds in [('train', train), ('val', val)]:
        ident = ds.identity()
        assert ident['split'] == split and not ident['contains_synthetic'] and not ident['manifest_contains_synthetic']
        assert len(ds) == PLAN['counts'][split]['knees']
        ids = {s.patient_id for s in ds._specs}; patients.append(ids)
        assert len(ids) == PLAN['counts'][split]['patients']
        assert len({s.film_id for s in ds._specs}) == PLAN['counts'][split]['films']
    assert not patients[0] & patients[1], 'PATIENT_SPLIT_OVERLAP'
    from collections import Counter
    actual_counts = Counter(int(s.kl) for s in train._specs)
    assert [actual_counts[i] for i in range(5)] == PLAN['train_class_counts'], 'TRAIN_CLASS_COUNTS_MISMATCH'
    if PLAN['weight_scheme'] == 'capped_sqrt':
        assert np.allclose(class_weights(PLAN['class_weight_reference_counts'], 3.0), PLAN['class_weights'], rtol=0, atol=1e-12)
    return train, val


def train(engine, counter, manager, run, sampler, resume):
    train_ds, val_ds = datasets()
    progress = initial_progress()
    if resume:
        progress, local, _ = manager.load(engine); rng_restore(local['rng'])
        assert evidence(engine, progress) == local['evidence'], 'FORMAL_RESUME_STATE_MISMATCH'
        if RANK == 0: write_json(run / 'epoch_history.json', progress['history'])
        publish_best(progress, run, manager)
    write_status(run, 'training', progress, sampler, {'dataset_identities': [train_ds.identity(), val_ds.identity()]})
    while progress['epoch'] < PLAN['epochs']:
        epoch = progress['epoch']
        order, order_hash = _order(len(train_ds), PLAN['seed'], epoch)
        progress['order_sha256'] = order_hash
        for b in rank_batches(order, rank=RANK, start_batch=progress['next_batch']):
            sampler.set_phase('train')
            t = time.perf_counter()
            x, y = real_batch(train_ds, b['indices'])
            CURRENT_BATCH.update(epoch=epoch, next_batch=b['batch_index'], last_successful_step=progress['step'])
            update = train_step(engine, counter, x, y, b['global_count'])
            torch.cuda.synchronize()
            # Use the slower rank's batch time so the global cursor and statistics match after resuming.
            elapsed = torch.tensor(time.perf_counter() - t, device=DEVICE, dtype=torch.float64)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            progress['train_seconds'] += float(elapsed)
            progress['step'] += 1; progress['next_batch'] = b['batch_index'] + 1
            progress['samples_seen'] += update['samples']; progress['epoch_samples'] += update['samples']
            progress['epoch_loss_sum'] += update['loss_sum']; progress['epoch_correct'] += update['correct']
            progress['epoch_ce_sum'] += update['ce_sum']; progress['epoch_rps_sum'] += update['rps_sum']
            progress['skipped_amp_attempts'] += update['skips']
            del x, y
            if progress['step'] % 10 == 0 or progress['step'] == 1:
                write_status(run, 'training', progress, sampler,
                    {'mean_update_seconds_with_io': progress['train_seconds'] / progress['step'],
                     'learning_rates': engine.lr_scheduler.get_last_lr()})
            if progress['step'] == PLAN['startup_checkpoint_step'] or progress['step'] in PLAN.get('early_verification_checkpoint_steps', []) or progress['step'] % PLAN['checkpoint_every_steps'] == 0:
                write_status(run, 'saving_checkpoint', progress, sampler)
                sampler.set_phase('checkpoint'); t = time.perf_counter()
                tag = f"s{progress['step']:06d}_e{epoch:02d}_b{progress['next_batch']:03d}"
                receipt = save_transaction(engine, manager, progress, tag)
                elapsed = torch.tensor(time.perf_counter() - t, device=DEVICE, dtype=torch.float64)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                progress['checkpoint_seconds'] += float(elapsed)
                write_status(run, 'training', progress, sampler, {'latest_checkpoint': receipt})
        assert progress['next_batch'] == PLAN['steps_per_epoch'] and progress['epoch_samples'] == len(train_ds)
        sampler.set_phase('validation'); write_status(run, 'validating', progress, sampler)
        t = time.perf_counter(); arrays, metrics = collect_predictions(engine, val_ds)
        elapsed = torch.tensor(time.perf_counter() - t, device=DEVICE, dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        progress['validation_seconds'] += float(elapsed)
        row = {'epoch': epoch + 1, 'step': progress['step'], 'train_cross_entropy': progress['epoch_ce_sum'] / len(train_ds),
               'train_objective': progress['epoch_loss_sum'] / len(train_ds),
               'train_ordinal_rps': progress['epoch_rps_sum'] / len(train_ds), 'train_samples': progress['epoch_samples'],
               'train_online_accuracy': progress['epoch_correct'] / len(train_ds), 'validation': metrics,
               'validation_seconds': float(elapsed)}
        progress['history'].append(row)
        key = [metrics['qwk'], -metrics['mae']]
        improved = metrics['qwk'] is not None and (progress['best_key'] is None or tuple(key) > tuple(progress['best_key']))
        next_tag = f"s{progress['step']:06d}_e{epoch+1:02d}_b000"
        if improved:
            progress.update(best_key=key, best_epoch=epoch + 1, best_metrics=metrics, best_tag=next_tag)
            delta_path = f'best_delta_epoch{epoch+1:02d}.pt'
            save_best_delta(engine, run / delta_path)
            box = [file_sha(run / delta_path) if RANK == 0 else None]
            dist.broadcast_object_list(box, src=0)
            progress.update(best_delta_path=delta_path, best_delta_sha256=box[0])
        if RANK == 0:
            np.savez_compressed(run / f'val_epoch{epoch+1:02d}.npz', **arrays)
            write_json(run / 'epoch_history.json', progress['history'])
        progress.update(epoch=epoch+1, next_batch=0, epoch_loss_sum=0.0, epoch_ce_sum=0.0, epoch_rps_sum=0.0, epoch_correct=0, epoch_samples=0)
        sampler.set_phase('checkpoint'); write_status(run, 'saving_checkpoint', progress, sampler)
        receipt = save_transaction(engine, manager, progress, next_tag)
        publish_best(progress, run, manager)
        write_status(run, 'epoch_completed', progress, sampler, {'latest_checkpoint': receipt})
    assert progress['step'] == PLAN['total_steps'] and progress['samples_seen'] == PLAN['counts']['train']['knees'] * PLAN['epochs']
    best_model = export_best(progress, run, manager)
    if RANK == 0:
        write_json(run / 'result.json', {'state': 'completed_train_val', 'progress': progress,
            'test_targets_accessed': False, 'clinical_validation': False, 'plan_sha256': file_sha(HERE / 'plan.json'),
            'final_checkpoint': receipt, 'best_model': best_model,
            'best_selection': json.loads((run/'best_selection.json').read_text()), 'completed_at': utc()})
    write_status(run, 'completed_train_val', progress, sampler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['preflight_save', 'preflight_resume', 'train'], required=True)
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    assert Path.cwd() == ROOT and int(os.environ['WORLD_SIZE']) == 2
    for p, version in PLAN['expected_versions'].items(): assert metadata.version(p) == version
    binding = {'plan_sha256': file_sha(HERE/'plan.json'), 'source_manifest_sha256': file_sha(HERE/'source_manifest.json')}
    if args.mode == 'train':
        auth = json.loads((HERE/'authorization.json').read_text())
        assert auth['binding'] == binding and auth['authorized'] is True
        assert auth['scope'] == PLAN['run_name']
    run = ROOT / PLAN['run_directory' if args.mode == 'train' else 'preflight_directory']
    run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(PLAN['cpu_threads_per_rank']); torch.set_num_interop_threads(PLAN['cpu_interop_threads']); torch.cuda.set_device(LOCAL_RANK)
    assert sorted(os.sched_getaffinity(0))==PLAN['cpu_affinity_logical_ids']
    write_json(run/f'cpu_policy_rank{RANK}.json',{'affinity':sorted(os.sched_getaffinity(0)),
        'torch_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
        'nice':os.getpriority(os.PRIO_PROCESS,0),'dataset_workers':PLAN['dataset_workers']})
    assert torch.cuda.is_bf16_supported(), 'BF16_UNSUPPORTED'
    assert 'GPU-' + str(torch.cuda.get_device_properties(LOCAL_RANK).uuid).removeprefix('GPU-') == PLAN['gpu_uuids'][RANK]
    random.seed(PLAN['seed']); np.random.seed(PLAN['seed']); torch.random.default_generator.manual_seed(PLAN['seed'])
    torch.cuda.manual_seed(PLAN['seed']); torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    deepspeed.init_distributed(dist_backend='nccl', timeout=dt.timedelta(minutes=10))
    sampler = BoardSampler.for_uuid(PLAN['gpu_uuids'][RANK]).start()
    passed = False
    try:
        write_status(run, 'initializing', initial_progress(), sampler)
        engine, counter = engine_init()
        engine._audit_run = run
        torch.cuda.manual_seed(PLAN['seed'] + 1009 * RANK)
        manager = Checkpoints(run/'checkpoints', PLAN['backup_directory'] + ('' if args.mode=='train' else '_preflight'), binding)
        if args.mode == 'train': train(engine, counter, manager, run, sampler, args.resume)
        else: preflight(engine, counter, manager, run, args.mode)
        passed = True; dist.barrier()
    except BaseException as exc:
        write_json(run/f'failure_rank{RANK}.json', {'at': utc(), 'type': type(exc).__name__, 'mode': args.mode,
                     'context': dict(CURRENT_BATCH), 'code': str(exc) if str(exc).isupper() and len(str(exc)) < 100 else 'SEE_SERVER_RAW_LOG'})
        traceback.print_exc()
    finally:
        write_json(run/f'{args.mode}_memory_rank{RANK}.json', {'passed': passed, 'board': sampler.stop(),
                    'allocated_peak_bytes': torch.cuda.max_memory_allocated(), 'reserved_peak_bytes': torch.cuda.max_memory_reserved()})
        dist.destroy_process_group()
    return 0 if passed else 2


if __name__ == '__main__': sys.exit(main())
