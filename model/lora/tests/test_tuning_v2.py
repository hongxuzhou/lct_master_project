"""Check the bounded search and reject incomplete or mixed-version selections."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import load_rows, read_protocol, select_train_subset, condition_rows
from train import training_plan
from tuning.plans import load_plan, resolve_assignment
from tuning.register_sweeps import configs
from tuning.run_sweep import training_args
from tuning.summarise import collect


class TuningV2Tests(unittest.TestCase):
    def test_exact_candidates_and_both_cluster_assignments(self):
        plan = load_plan('lr-rank-v2')
        expected = {(lr, r) for lr in (1e-4, 2e-4, 3e-4) for r in (16, 32)}
        expected |= {(4e-4, 32), (3e-4, 64), (4e-4, 64)}
        self.assertEqual(set(plan['candidates'].values()), expected)
        with patch.dict(os.environ, DATASET_DIR='/tmp/data', MODEL_CACHE='/tmp/model'):
            for size in ('2B', '9B', '27B'):
                for condition, spec in configs(size, 'lr-rank-v2').items():
                    self.assertEqual(spec['run_cap'], 9)
                    self.assertEqual(set(spec['parameters']['candidate']['values']), set(plan['candidates']))
                    self.assertNotIn('learning_rate', spec['parameters'])
                    self.assertIn('--plan', spec['command'])
                    for candidate in plan['candidates']:
                        assignment = dict(candidate=candidate, model_size=size, condition=condition, seed=17)
                        for cluster in ('habrok', 'lst'):
                            args = training_args(assignment, Path('/tmp/run'), 'lr-rank-v2', cluster)
                            self.assertEqual(args.cluster, cluster)
                            self.assertEqual(args.protocol, plan['protocol'])
                            self.assertEqual(args.train_subset, plan['subset'])
                            self.assertEqual(args.lora_alpha, 2 * args.lora_rank)

    def test_invalid_or_conflicting_assignment_is_rejected(self):
        plan = load_plan('lr-rank-v2')
        for assignment in [dict(candidate='lr4e-4-r16', seed=17),
                           dict(candidate='lr4e-4-r64', seed=17, lora_rank=32),
                           dict(candidate='lr4e-4-r64', seed=18)]:
            with self.assertRaises(ValueError):
                resolve_assignment(assignment, plan)

    def test_actual_subset_and_updated_schedules(self):
        plan = load_plan('lr-rank-v2')
        protocol = read_protocol(plan['protocol'])
        dataset = Path(__file__).resolve().parents[3] / 'repair_data_gen/training_dataset'
        rows, _ = select_train_subset(load_rows(dataset, protocol), plan['subset'], protocol)
        old = load_plan()['subset']
        self.assertTrue(set(json.loads(old.read_text())['example_ids']) <
                        set(json.loads(plan['subset'].read_text())['example_ids']))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            select_train_subset(rows, old, protocol)
        with patch.dict(os.environ, DATASET_DIR=str(dataset), MODEL_CACHE='/tmp/model'):
            for c in ('C1', 'C2', 'C3'):
                args = training_args(dict(candidate='lr4e-4-r64', model_size='2B', condition=c, seed=17),
                                     Path('/tmp/run'), 'lr-rank-v2')
                n = len(condition_rows(rows['train'], c))
                self.assertEqual(n, 969 if c == 'C1' else 1423)
                result = training_plan(n, protocol, args)
                self.assertEqual((result['total_steps'], result['warmup_steps'], result['eval_every']),
                                 (183, 6, 19) if c == 'C1' else (267, 9, 27))

    def test_selection_rejects_missing_duplicate_and_mixed_provenance(self):
        plan = load_plan('lr-rank-v2')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, (candidate, (lr, rank)) in enumerate(plan['candidates'].items()):
                path = root / '2B' / 'C3' / str(i)
                path.mkdir(parents=True)
                config = dict(purpose='tuning', model_size='2B', condition='C3', seed=17,
                              learning_rate=lr, lora_rank=rank, wandb_run_id=str(i),
                              subset={'manifest_sha256': hashlib.sha256(plan['subset'].read_bytes()).hexdigest()},
                              code_sha256={'protocol.json': hashlib.sha256(plan['protocol'].read_bytes()).hexdigest(),
                                           'train.py': 't', 'data.py': 'd'},
                              sweep_assignment={'tuning_plan': 'lr-rank-v2', 'candidate': candidate,
                                                'grid_sha256': plan['sha256'], 'plan_code_sha256': 'pc',
                                                'sweep_entry_sha256': 'entry'})
                (path / 'config.json').write_text(json.dumps(config))
                (path / 'summary.json').write_text(json.dumps(dict(status='development_passed',
                    best_eval_loss=.1+i*.01, reload_eval_loss=.1+i*.01)))
            def result():
                return collect(root, 'lr-rank-v2')['groups']['2B/C3']
            self.assertEqual(result()['selected_run'], '0')
            failure = root / '2B/C3/0/failure.json'
            failure.write_text('{}')
            self.assertIsNone(result()['selected_run'])
            self.assertEqual(result()['missing'], [[1e-4, 16]])
            failure.unlink()
            source = root / '2B/C3/0/config.json'
            original = source.read_text()
            config = json.loads(original)
            config['subset']['manifest_sha256'] = 'old-data'
            source.write_text(json.dumps(config))
            self.assertFalse(result()['matches_plan'])
            self.assertIsNone(result()['selected_run'])
            source.write_text(original)
            duplicate = root / '2B/C3/duplicate'
            duplicate.mkdir()
            (duplicate / 'config.json').write_text(original)
            (duplicate / 'summary.json').write_text((source.parent / 'summary.json').read_text())
            self.assertTrue(result()['duplicate_configurations'])
            self.assertIsNone(result()['selected_run'])
            self.assertIsNone(collect(root)['groups']['2B/C3']['selected_run'])


if __name__ == '__main__':
    unittest.main()
