"""Checks for shared inputs, checkpoint relocation and scope evidence."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tarfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from adapters import prepare_adapter
from panel import HERE, digest, read_panel
from run import checkpoint_spec
from review import graph_evidence
from build_bundle import build


class ProbeTests(unittest.TestCase):
    def test_deployment_bundle_runs_without_full_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / 'bundle.tar.gz'
            build(archive_path)
            with tarfile.open(archive_path) as archive:
                archive.extractall(root / 'unpacked')
            bundle = root / 'unpacked/repairsbn-restart-probe-v1'
            inventory = json.loads((bundle / 'bundle.json').read_text())
            for name, sha in inventory['files'].items():
                self.assertEqual(digest(bundle / name), sha)
            frozen = json.loads((bundle / 'frozen-9B-C3.json').read_text())
            self.assertEqual({(r['model_size'], r['condition'], int(r['seed'])) for r in frozen},
                             {('9B', 'C3', s) for s in (24, 42, 4224)})
            probe = bundle / 'evaluation/probing/tea_coffee_v1'
            result = subprocess.run([sys.executable, str(probe / 'run.py'), '--stimuli-only'],
                                    cwd=root, capture_output=True, text=True, check=True)
            self.assertIn('16 unique inputs', result.stdout)
            subprocess.run(['bash', '-n', str(probe / 'lst_worker.sh')], check=True)
            subprocess.run([sys.executable, str(probe / 'lst_preflight.py'), '--help'],
                           capture_output=True, check=True)

    def test_download_missing_then_reuse_offline_and_preserve_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.write_bytes(b'fixture weights')
            config = root / 'config'
            config.write_bytes(b'fixture config')
            run_id = 'formal-habrok-9B-C3-s24-fixture'
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps([dict(model_size='9B', condition='C3', seed='24',
                run_id=run_id, best_step='1602', config_sha256=digest(config),
                files={'adapter_model.safetensors': digest(source)})]))
            calls = []
            revision = 'a' * 40

            def download(**kw):
                calls.append(kw['filename'])
                self.assertEqual(kw['revision'], revision)
                path = Path(kw['local_dir']) / kw['filename']
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(config.read_bytes() if path.name == 'config.json' else source.read_bytes())

            hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=lambda repo: SimpleNamespace(sha=revision)),
                                  hf_hub_download=download)
            store = root / 'store'
            with patch.dict(sys.modules, huggingface_hub=hub):
                checkpoint, _, meta = prepare_adapter(manifest, 24, store, 'owner/adapters')
                self.assertEqual(len(calls), 2)
                self.assertEqual(meta['revision'], revision)
                # A complete local store needs neither authentication nor Hub access.
                with patch.dict(sys.modules, huggingface_hub=None):
                    _, _, reused = prepare_adapter(manifest, 24, store, 'owner/adapters', allow_download=False)
                    self.assertEqual(reused['downloaded_files'], [])
                weights = checkpoint / 'adapter_model.safetensors'
                weights.unlink()
                prepare_adapter(manifest, 24, store, 'owner/adapters')
                self.assertEqual(len(calls), 3)
                weights.write_bytes(b'corrupt but preserved')
                with self.assertRaisesRegex(ValueError, 'Existing adapter file hash mismatch'):
                    prepare_adapter(manifest, 24, store, 'owner/adapters')
                self.assertEqual(weights.read_bytes(), b'corrupt but preserved')
                self.assertEqual(len(calls), 3)

    def test_tree_and_symbol_updates(self):
        panel, rows = read_panel()
        by_id = {r['id']: r for r in rows}
        self.assertEqual(by_id['U01']['paths'], ['unmarked_ambiguous', 'listing', 'listing_oxford'])
        self.assertEqual(by_id['R02']['text'], 'I ordered tea, no,')
        self.assertEqual(by_id['O01']['added'], ',')
        self.assertEqual(len(panel['paths']), 5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'stimuli.json'
            panel['checkpoints'][6]['text'] += '.'
            path.write_text(json.dumps(panel))
            with self.assertRaisesRegex(ValueError, 'cumulative update'):
                read_panel(path)

    def test_relocated_checkpoint_keeps_hash_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'copied-adapter'
            checkpoint.mkdir()
            weights = checkpoint / 'adapter_model.safetensors'
            weights.write_bytes(b'fixture only')
            config_path = root / 'original-config.json'
            config_path.write_text(json.dumps(dict(run_id='original', model_size='9B', condition='C3',
                                                  seed=24, protocol={'models': {'9B': {'id': 'base', 'revision': 'rev'}}})))
            manifest = root / 'frozen.json'
            manifest.write_text(json.dumps([dict(run_id='original', model_size='9B', condition='C3', seed='24',
                                                  base_model='base', base_revision='rev',
                                                  checkpoint_path='/old/cluster/path',
                                                  config_sha256=digest(config_path),
                                                  files={weights.name: digest(weights)})]))
            _, path, _, _ = checkpoint_spec(manifest, 24, checkpoint, config_path)
            self.assertEqual(path, checkpoint)
            weights.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Checkpoint hash mismatch'):
                checkpoint_spec(manifest, 24, checkpoint, config_path)

    def test_scope_and_partial_run_review(self):
        sbn = ('female.n.02 order.v.01 Agent -1 CORRECTION <1 tea.n.01 ThemeOf -1 '
               'CONJUNCTION <2 coffee.n.01 ThemeOf -2')
        ev = graph_evidence(sbn)
        self.assertEqual(ev['pred_status'], 'valid')
        self.assertEqual(len(ev['cc_pairs']), 1)
        coffee = next(n['id'] for n in ev['nodes'] if n['label'] == 'coffee.n.01')
        relation = next(r for r in ev['relations'] if r['target'] == coffee)
        self.assertNotEqual(relation['written_scope'], relation['effective_scope'])
        bad = graph_evidence('order.v.01 CORRECTION <1 tea.n.01 ThemeOf -99')
        self.assertEqual(bad['pred_status'], 'invalid')
        self.assertIsNone(bad['cc_pairs'])
        self.assertEqual(bad['raw_correction_count'], 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stimuli = root / 'stimuli.json'
            stimuli.write_bytes((HERE / 'stimuli.json').read_bytes())
            (root / 'run.json').write_text(json.dumps(dict(status='failed', seed=24,
                stimuli_sha256=digest(stimuli), checkpoint={'run_id': 'fixture'})))
            (root / 'predictions.jsonl').write_text(json.dumps(dict(id='S01', text='I ordered tea',
                seed=24, condition='C3', run_id='fixture', pred_sbn=sbn, tokens=20, ended_with_eos=True)) + '\n')
            output = root / 'review'
            subprocess.run([sys.executable, str(HERE / 'review.py'), '--run', str(root),
                            '--output', str(output)], check=True, capture_output=True, text=True)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['counts'], {'valid': 1, 'missing': 15})
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run([sys.executable, str(HERE / 'review.py'), '--run', str(root),
                                '--output', str(output)], check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
