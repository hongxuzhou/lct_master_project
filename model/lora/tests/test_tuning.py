"""CPU checks for Sweep assignment, schedule and run ownership."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import read_protocol, select_train_subset, digest
from train import Recorder, training_plan, expected_lora_parameters
from tuning.register_sweeps import configs
from tuning.run_sweep import training_args, SUBSET, PROTOCOL
from tuning.summarise import collect


class TuningTests(unittest.TestCase):
    def test_grid_assignment_and_schedule(self):
        protocol = read_protocol(PROTOCOL)
        with patch.dict(os.environ, {'DATASET_DIR': '/tmp/data', 'MODEL_CACHE': '/tmp/model'}):
            for condition, spec in configs('2B').items():
                self.assertEqual(spec['run_cap'], 6)
                for lr in spec['parameters']['learning_rate']['values']:
                    for rank in spec['parameters']['lora_rank']['values']:
                        args = training_args(dict(model_size='2B', condition=condition,
                            learning_rate=lr, lora_rank=rank, seed=17), Path('/tmp/run'))
                        self.assertEqual(args.lora_alpha, 2*rank)
                        self.assertEqual(args.generation_examples, 0)
                        plan = training_plan(949 if condition == 'C1' else 1403, protocol, args)
                        self.assertEqual((plan['total_steps'], plan['warmup_steps'], plan['eval_every']),
                                         (180, 6, 18) if condition == 'C1' else (264, 8, 27))

    def test_lora_dimensions_scale_with_rank(self):
        candidates = [('a', SimpleNamespace(in_features=10, out_features=20)),
                      ('b', SimpleNamespace(in_features=20, out_features=5))]
        self.assertEqual(expected_lora_parameters(candidates, 16), 880)
        self.assertEqual(expected_lora_parameters(candidates, 32), 1760)

    def test_external_run_is_not_initialised_or_finished_by_recorder(self):
        run = SimpleNamespace(config={}, summary={}, log=lambda *a, **k: None)
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory), {'learning_rate': 1e-4}, 'online', {}, external_run=run)
            recorder.log('eval', 18, best_eval_loss=0.5)
            recorder.summary({'best_checkpoint': 'checkpoint-18'})
            recorder.close(0)
        self.assertEqual(run.config['learning_rate'], 1e-4)
        self.assertEqual(run.summary['best_checkpoint'], 'checkpoint-18')

    def test_subset_rejects_tampering_and_keeps_validation(self):
        protocol = read_protocol(PROTOCOL)
        manifest = json.loads(SUBSET.read_text())
        rows = {'train': [{'example_id': 'f', 'variant': 'pmb_fluent', 'item_id': None},
                          {'example_id': 'b', 'variant': 'rsr_base', 'item_id': 'i'},
                          {'example_id': 'm', 'variant': 'rsr_editing_expression', 'item_id': 'i'}],
                'validation': [object()]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            manifest['example_ids'] = ['f', 'b', 'm']
            manifest['example_ids_sha256'] = digest(manifest['example_ids'])
            path.write_text(json.dumps(manifest))
            selected, _ = select_train_subset(rows, path, protocol)
            self.assertIs(selected['validation'], rows['validation'])
            manifest['example_ids'] = ['f', 'b']
            manifest['example_ids_sha256'] = digest(manifest['example_ids'])
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'splits'):
                select_train_subset(rows, path, protocol)

    def test_summary_requires_complete_grid_and_excludes_failed_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, (lr, rank) in enumerate(( (lr, rank) for lr in (1e-4, 2e-4, 3e-4) for rank in (16, 32))):
                path = root/'2B'/'C3'/str(i)
                path.mkdir(parents=True)
                config = dict(purpose='tuning', model_size='2B', condition='C3',
                    learning_rate=lr, lora_rank=rank, seed=17, wandb_run_id=str(i),
                    subset={'manifest_sha256': 'fixed'},
                    code_sha256={'protocol.json': 'p', 'train.py': 't', 'data.py': 'd'},
                    sweep_assignment={'sweep_entry_sha256': 's'})
                (path/'config.json').write_text(json.dumps(config))
                (path/'summary.json').write_text(json.dumps(dict(status='development_passed',
                    best_eval_loss=0.1+i*0.01, reload_eval_loss=0.1+i*0.01)))
            self.assertEqual(collect(root)['groups']['2B/C3']['selected_run'], '0')
            (root/'2B/C3/0/failure.json').write_text('{}')
            result = collect(root)
            self.assertFalse(result['groups']['2B/C3']['complete_grid'])
            self.assertIsNone(result['groups']['2B/C3']['selected_run'])
            self.assertEqual(len(result['incomplete_or_failed']), 1)


if __name__ == '__main__':
    unittest.main()
