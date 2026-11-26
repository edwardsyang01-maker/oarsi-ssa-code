import json
import os
from pathlib import Path
import shutil
import torch
import torch.distributed as dist
from model_runtime import file_sha, write_json, utc


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def torch_save_atomic(value, path):
    tmp = path.with_suffix(path.suffix + '.partial')
    with tmp.open('wb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


class Checkpoints:
    def __init__(self, local_root, nas_root, binding, keep=3):
        self.root, self.nas = Path(local_root), Path(nas_root)
        self.binding, self.keep = binding, keep
        self.root.mkdir(parents=True, exist_ok=True)
        if dist.get_rank() == 0: self.nas.mkdir(parents=True, exist_ok=True)
        dist.barrier()

    def save(self, engine, progress, rank_state, tag):
        rank = dist.get_rank()
        staging = self.root / ('.pending_' + tag)
        final = self.root / tag
        if rank == 0:
            assert not staging.exists() and not final.exists(), 'CHECKPOINT_TAG_ALREADY_EXISTS'
            staging.mkdir()
        dist.barrier()
        ok = engine.save_checkpoint(str(staging), tag='shards',
                                   client_state={'progress': progress, 'binding': self.binding},
                                   exclude_frozen_parameters=False)
        if ok is False: raise RuntimeError('DEEPSPEED_CHECKPOINT_SAVE_FAILED')
        torch_save_atomic(rank_state, staging / f'rank{rank}_state.pt')
        dist.barrier()
        receipt = None
        if rank == 0:
            files = []
            for p in sorted(staging.rglob('*')):
                if not p.is_file(): continue
                with p.open('rb') as f: os.fsync(f.fileno())
                files.append({'path': str(p.relative_to(staging)), 'bytes': p.stat().st_size, 'sha256': file_sha(p)})
            assert any(x['path'].endswith('optim_states.pt') for x in files)
            assert all((staging / f'rank{i}_state.pt').exists() for i in [0, 1])
            receipt = {'tag': tag, 'binding': self.binding, 'step': progress['step'],
                       'epoch': progress['epoch'], 'next_batch': progress['next_batch'],
                       'files': files, 'created_at': utc()}
            write_json(staging / 'manifest.json', receipt)
            sync_directory(staging)
            os.rename(staging, final); sync_directory(self.root)
            nas_pending = self.nas / ('.pending_' + tag)
            nas_final = self.nas / tag
            assert not nas_pending.exists() and not nas_final.exists()
            shutil.copytree(final, nas_pending)
            for x in files:
                target = nas_pending / x['path']
                with target.open('rb') as f: os.fsync(f.fileno())
                assert target.stat().st_size == x['bytes'] and file_sha(target) == x['sha256'], 'NAS_HASH_MISMATCH'
            assert file_sha(nas_pending / 'manifest.json') == file_sha(final / 'manifest.json')
            sync_directory(nas_pending)
            os.rename(nas_pending, nas_final); sync_directory(self.nas)
            receipt['nas_verified_at'] = utc()
            receipt['manifest_sha256'] = file_sha(final / 'manifest.json')
            receipt['total_bytes'] = sum(x['bytes'] for x in files)
            write_json(self.root / f'{tag}.commit.json', receipt)
            write_json(self.nas / f'{tag}.commit.json', receipt)
            write_json(self.root / 'latest.json', receipt)
            write_json(self.nas / 'latest.json', receipt)
            for journal_root in [self.root, self.nas]:
                with (journal_root / 'commit_history.jsonl').open('a') as journal:
                    journal.write(json.dumps(receipt, ensure_ascii=False) + '\n')
                    journal.flush(); os.fsync(journal.fileno())
            sync_directory(self.root); sync_directory(self.nas)
            self.prune(progress.get('best_tag'))
        box = [receipt]; dist.broadcast_object_list(box, src=0)
        return box[0]

    def prune(self, best_tag):
        committed = []
        for p in self.root.glob('s*.commit.json'):
            obj = json.loads(p.read_text()); committed.append((obj['created_at'], obj['tag']))
        committed.sort()
        keep = {tag for _, tag in committed[-self.keep:]}
        if best_tag: keep.add(best_tag)
        for _, tag in committed:
            if tag in keep: continue
            assert tag.startswith('s') and '/' not in tag and (self.nas / f'{tag}.commit.json').exists()
            for root in [self.root, self.nas]:
                shutil.rmtree(root / tag)
                (root / f'{tag}.commit.json').unlink()

    def load(self, engine, tag=None):
        receipt = json.loads((self.root / ('latest.json' if tag is None else f'{tag}.commit.json')).read_text())
        assert receipt['binding'] == self.binding, 'CHECKPOINT_BINDING_MISMATCH'
        directory = self.root / receipt['tag']
        assert file_sha(directory / 'manifest.json') == receipt['manifest_sha256']
        manifest = json.loads((directory / 'manifest.json').read_text())
        for x in manifest['files']:
            p = directory / x['path']
            assert p.stat().st_size == x['bytes'] and file_sha(p) == x['sha256'], 'LOCAL_CHECKPOINT_CORRUPT'
        path, state = engine.load_checkpoint(str(directory), tag='shards', load_optimizer_states=True,
                                             load_lr_scheduler_states=True)
        assert path and state['binding'] == self.binding
        local = torch.load(directory / f'rank{dist.get_rank()}_state.pt', map_location='cpu', weights_only=False)
        assert local['progress'] == state['progress'], 'RANK_CURSOR_MISMATCH'
        return state['progress'], local, receipt
