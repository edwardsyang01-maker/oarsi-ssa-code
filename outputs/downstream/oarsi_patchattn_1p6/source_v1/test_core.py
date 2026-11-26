import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from core import (TARGETS, SWAP, PATCHES, GROUPS, lambda_at, load_targets, canonical_flip, random_flip, oarsi_loss,
                  PatchPool, AuxHeads, extra_parameters, qwk, auc, flip_consistency)
import control


class CoreTests(unittest.TestCase):
    def test_lambda_ramp(self):
        self.assertEqual([lambda_at(2.0, e) for e in range(1, 7)], [0.5, 1.0, 1.5, 2.0, 2.0, 2.0])

    def test_swap_is_medial_lateral_involution(self):
        self.assertEqual([SWAP[s] for s in SWAP], list(range(6)))
        for i, j in enumerate(SWAP):
            a, b = TARGETS[i], TARGETS[j]
            self.assertEqual(a[:-1], b[:-1]); self.assertNotEqual(a[-1], b[-1])
            self.assertEqual({a[-1], b[-1]}, {'m', 'l'})

    def test_canonical_flip(self):
        x = torch.arange(2*1*3*4, dtype=torch.float32).view(2, 1, 3, 4)
        original = x.clone()
        y = canonical_flip(x, torch.tensor([False, True]))
        self.assertTrue(torch.equal(y[0], x[0])); self.assertTrue(torch.equal(y[1], x[1].flip(-1)))
        self.assertTrue(torch.equal(canonical_flip(y, torch.tensor([False, True])), x))
        self.assertTrue(torch.equal(x, original))

    def test_random_flip_swaps_only_flipped_rows(self):
        x = torch.rand(3, 1, 4, 4)
        t = torch.tensor([[0, 1, 2, 3, 1, 0], [3, 2, 1, 0, -1, 2], [1, -1, 0, 2, 3, 1]])
        mask = torch.tensor([True, False, True])
        img, out = random_flip(x, t, mask, swap=True)
        self.assertTrue(torch.equal(img[0], x[0].flip(-1)) and torch.equal(img[1], x[1]))
        self.assertEqual(out[0].tolist(), [1, 0, 3, 2, 0, 1])
        self.assertEqual(out[1].tolist(), t[1].tolist())
        self.assertEqual(out[2].tolist(), [-1, 1, 2, 0, 1, 3])
        img2, out2 = random_flip(x, t, mask, swap=False)
        self.assertTrue(torch.equal(img2, img)); self.assertTrue(torch.equal(out2, t))

    def test_swap_keeps_label_with_image_side(self):
        x = torch.zeros(1, 1, 4, 8); x[..., 6:] = 1.0
        t = torch.tensor([[2, 0, 2, 0, 2, 0]])
        img, out = random_flip(x, t, torch.tensor([True]), swap=True)
        self.assertEqual(float(img[..., 6:].sum()), 0.0)
        self.assertEqual(out[0, 0].item(), 0)
        self.assertEqual(out[0, 1].item(), 2)

    def test_oarsi_loss_masked_mean(self):
        torch.manual_seed(3)
        logits = torch.randn(3, 6, 4, requires_grad=True)
        t = torch.tensor([[0, 1, 2, 3, 0, 1], [-1, -1, 2, -1, -1, 0], [-1]*6])
        got = oarsi_loss(logits, t)
        expected = []
        for i in range(3):
            terms = [F.cross_entropy(logits[i, j][None], t[i, j][None]) for j in range(6) if t[i, j] >= 0]
            expected.append(torch.stack(terms).mean() if terms else logits.sum()*0)
        torch.testing.assert_close(got, torch.stack(expected))
        got.sum().backward()
        self.assertTrue(torch.all(logits.grad[2] == 0))
        self.assertTrue(torch.all(logits.grad[1, [0, 1, 3, 4]] == 0))
        self.assertTrue(torch.any(logits.grad[1, 2] != 0))

    def test_aux_heads_shapes_counts_and_init(self):
        state = torch.get_rng_state()
        heads = {g: AuxHeads(g) for g in GROUPS if g != 'P0'}
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        self.assertEqual(extra_parameters('P0'), 0)
        cls, patch_ = torch.randn(2, 1024), torch.randn(2, PATCHES, 1024)
        for g, h in heads.items():
            self.assertEqual(sum(p.numel() for p in h.parameters()), extra_parameters(g))
            out = h(cls, patch_, return_attention=True)
            cfg = GROUPS[g]
            if cfg['pool']:
                self.assertEqual(tuple(out['kl_extra'].shape), (2, 5))
                self.assertTrue(torch.all(out['kl_extra'] == 0))
                k = 7 if cfg['oarsi'] == 'pool' else 1
                self.assertEqual(tuple(out['attention'].shape), (2, k, PATCHES))
                torch.testing.assert_close(out['attention'].sum(-1), torch.ones(2, k))
            else:
                self.assertIsNone(out['kl_extra'])
            if cfg['oarsi']:
                self.assertEqual(tuple(out['oarsi'].shape), (2, 6, 4))
            else:
                self.assertIsNone(out['oarsi'])
        torch.testing.assert_close(heads['P1'].pool.query[0], heads['P3'].pool.query[0])
        for (k1, v1), (k2, v2) in zip(heads['P3'].state_dict().items(), heads['P4'].state_dict().items()):
            self.assertEqual(k1, k2); self.assertTrue(torch.equal(v1, v2))

    def test_patch_pool_uses_position(self):
        pool = PatchPool(1)
        with torch.no_grad():
            pool.query.normal_(0, 5.0)
            pool.position.normal_(0, 1.0)
        torch.manual_seed(5)
        content = torch.randn(1, PATCHES, 1024)
        shifted = content.roll(28, dims=1)
        _, a = pool(content, True); _, b = pool(shifted, True)
        self.assertGreater(float((a.roll(28, dims=-1)-b).abs().max()), 1e-4)
        with torch.no_grad():
            pool.position.zero_()
        _, a0 = pool(content, True); _, b0 = pool(shifted, True)
        torch.testing.assert_close(a0.roll(28, dims=-1), b0)

    def test_qwk_and_auc(self):
        self.assertAlmostEqual(qwk([0, 1, 2, 3], [0, 1, 2, 3], 4), 1.0)
        y, p = np.array([0, 1, 2, 3, 1, 2]), np.array([0, 2, 2, 1, 1, 3])
        o = np.zeros((4, 4)); [o.__setitem__((a, b), o[a, b]+1) for a, b in zip(y, p)]
        w = np.array([[(i-j)**2/9 for j in range(4)] for i in range(4)])
        e = np.outer(o.sum(1), o.sum(0))/o.sum()
        self.assertAlmostEqual(qwk(y, p, 4), 1-(w*o).sum()/(w*e).sum())
        s = np.array([.1, .4, .4, .8, .3, .4]); pos = np.array([0, 1, 0, 1, 0, 1], bool)
        pairs = [(1.0 if a > b else .5 if a == b else 0.0) for a, b in itertools.product(s[pos], s[~pos])]
        self.assertAlmostEqual(auc(s, pos), float(np.mean(pairs)))
        self.assertIsNone(auc(s, np.zeros(6, bool)))

    def test_flip_consistency(self):
        p = np.eye(4)[np.array([[0, 1, 2, 3, 0, 1]])]
        self.assertEqual(flip_consistency(p, p)['same'], 1.0)
        self.assertEqual(flip_consistency(p, p[:, SWAP])['swapped'], 1.0)

    def test_load_targets_alignment(self):
        rows = [{'key': 'a', 'split': 'train', 'kl': 1, 'clinical_side': 'R', 'visit': 'V00'},
                {'key': 'b', 'split': 'train', 'kl': 3, 'clinical_side': 'L', 'visit': 'V06'}]
        labels = [{**rows[0], **{n: None for n in TARGETS}, 'jsm': 1, 'jsl': 0},
                  {**rows[1], 'osfm': 2, 'osfl': 1, 'ostm': 3, 'ostl': 0, 'jsm': 2, 'jsl': 1}]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'x.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in labels))
            out = load_targets(rows, path, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(out.tolist(), [[-1, -1, -1, -1, 1, 0], [2, 1, 3, 0, 2, 1]])
            bad = [dict(rows[0], kl=2), rows[1]]
            with self.assertRaises(AssertionError):
                load_targets(bad, path)

    def test_flip_commutes_with_photometric_augmentation(self):
        spec = json.loads((HERE/'spec.json').read_text())
        sys.path.insert(0, str(Path(spec['project'])/spec['base_source']))
        try:
            from augmentation_policy import augment_knee
        except ImportError:
            self.skipTest('base augmentation unavailable')
        params = {'horizontal_flip_probability': 0.0, 'rotation_degrees': [0, 0], 'translation_fraction': [0, 0],
                  'gaussian_blur_probability': 0.0}
        x = torch.rand(4, 1, 16, 16)
        torch.manual_seed(11); a = augment_knee(x.flip(-1), params)
        torch.manual_seed(11); b = augment_knee(x, params).flip(-1)
        torch.testing.assert_close(a, b)

    def test_lambda_lock_formula_and_single_use(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d); (out/'calibration').mkdir()
            spec_sha = hashlib.sha256((HERE/'spec.json').read_bytes()).hexdigest()
            for g, ratios in {'P2': [2.0, 4.0, 6.0]*9, 'P3': [1.0, 3.0]*14}.items():
                value = control.SPEC['lambda_target']*statistics.median(ratios)
                for r in (0, 1):
                    (out/'calibration'/f'calibration_{g}_rank{r}.json').write_text(json.dumps({
                        'group': g, 'lambda': value, 'spec_sha256': spec_sha, 'valid_batches': len(ratios),
                        'real_optimizer_steps': 0, 'model_optimizer_rng_unchanged': True,
                        'all_ratios_main_over_oarsi': ratios}))
            with patch.object(control, 'OUTPUT', out):
                control.lock_lambda()
                lock = json.loads((out/control.SPEC['lambda_lock']).read_text())
                self.assertAlmostEqual(lock['lambda']['P2'], 0.25*4.0)
                self.assertAlmostEqual(lock['lambda']['P3'], 0.25*2.0)
                with self.assertRaises(AssertionError):
                    control.lock_lambda()


if __name__ == '__main__':
    unittest.main(verbosity=2)
