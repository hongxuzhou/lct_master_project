"""Resource routing, registry isolation and preview-only wrapper checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tuning.plans import load_plan
from tuning.registry import sweep_path
from tuning.submit import command
from tuning.lst.build_round_v2 import build_round

LORA = Path(__file__).resolve().parents[1]


class SubmissionTests(unittest.TestCase):
    def registry(self, directory, size):
        plan = load_plan('lr-rank-v2')
        spec = dict(tuning_plan=plan['name'], model_size=size, grid_sha256=plan['sha256'],
                    base_url='https://api.forge.coreweave.com',
                    subset_manifest_sha256=hashlib.sha256(plan['subset'].read_bytes()).hexdigest(),
                    protocol_sha256=hashlib.sha256(plan['protocol'].read_bytes()).hexdigest(),
                    sweeps={c: f'hxzhou-research/qwen35-rsr-lora/test1234{c}' for c in ('C1', 'C2', 'C3')})
        path = Path(directory) / 'registry.json'
        path.write_text(json.dumps(spec))
        return path

    def test_registry_rejects_wrong_size_and_old_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.registry(directory, '27B')
            self.assertTrue(sweep_path(path, '27B', 'C3').endswith('test1234C3'))
            with self.assertRaises(ValueError):
                sweep_path(path, '9B', 'C3')
            spec = json.loads(path.read_text())
            spec['subset_manifest_sha256'] = 'old'
            path.write_text(json.dumps(spec))
            with self.assertRaises(ValueError):
                sweep_path(path, '27B', 'C3')

    def test_model_gpu_routing_and_one_trial_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            for cluster, size in [('habrok', '2B'), ('lst', '9B'), ('lst', '27B')]:
                registry = self.registry(directory, size)
                cmd = command(cluster, size, registry, 'C3', 1)
                if cluster == 'habrok':
                    self.assertEqual(cmd[0], 'sbatch')
                    self.assertEqual(cmd[-3:], ['C3', '1', 'tuning'])
                else:
                    self.assertEqual(cmd[0], 'condor_submit')
                    self.assertIn('COUNT = 1', cmd)
                    self.assertIn('CAPABILITY = 9.0' if size == '27B' else 'CAPABILITY = 8.0', cmd)
                    self.assertIn('MIN_GPU_MB = 79000' if size == '27B' else 'MIN_GPU_MB = 39000', cmd)
                    self.assertIn('MAX_GPU_MB = 85000' if size == '27B' else 'MAX_GPU_MB = 41000', cmd)
            with self.assertRaises(ValueError):
                command('habrok', '27B', None, 'C3', 1, fit=True)
            with self.assertRaises(ValueError):
                command('lst', '9B', None, 'C3', 10, fit=True)

    def test_wrappers_preview_without_scheduler_or_gpu(self):
        env = {**os.environ, 'PYTHON': sys.executable}
        for cluster, extra in [('habrok', []), ('lst', ['--model-size', '9B']),
                               ('lst', ['--model-size', '27B'])]:
            result = subprocess.run(['bash', str(LORA / f'tuning/{cluster}/submit_v2.sh'),
                                     '--condition', 'C3', '--fit-check', *extra],
                                    env=env, text=True, capture_output=True, check=True)
            self.assertIn('sbatch' if cluster == 'habrok' else 'condor_submit', result.stdout)
            self.assertIn('fit', result.stdout)

    def test_default_previews_all_three_conditions_with_nine_trials(self):
        env = {**os.environ, 'PYTHON': sys.executable}
        with tempfile.TemporaryDirectory() as directory:
            for cluster, size in [('habrok', '2B'), ('lst', '9B'), ('lst', '27B')]:
                registry = self.registry(directory, size)
                args = ['bash', str(LORA / f'tuning/{cluster}/submit_v2.sh'), '--registry', str(registry)]
                if cluster == 'lst':
                    args += ['--model-size', size]
                result = subprocess.run(args, env=env, text=True, capture_output=True, check=True)
                lines = result.stdout.strip().splitlines()
                self.assertEqual(len(lines), 3)
                for c, line in zip(('C1', 'C2', 'C3'), lines):
                    self.assertIn(f'{c} 9 tuning' if cluster == 'habrok' else f'CONDITION = {c}', line)
                    if cluster == 'lst':
                        self.assertIn('COUNT = 9', line)
                        self.assertIn('WALL_SECONDS = 259200' if size == '27B' else 'WALL_SECONDS = 86400', line)

    def test_shell_syntax(self):
        scripts = list((LORA / 'tuning').rglob('*v2.sh'))
        scripts.append(LORA / 'tuning/habrok/agent_v2.sbatch')
        for path in scripts:
            subprocess.run(['bash', '-n', str(path)], capture_output=True, check=True)

    def test_lst_round_prepares_before_all_six_agents(self):
        with tempfile.TemporaryDirectory() as directory:
            dag = build_round(directory)
            text = dag.read_text()
            self.assertEqual(sum(line.startswith('JOB ') for line in text.splitlines()), 7)
            self.assertIn('PARENT PREPARE CHILD TUNE_9B_C1 TUNE_9B_C2 TUNE_9B_C3 TUNE_27B_C1 TUNE_27B_C2 TUNE_27B_C3', text)
            self.assertNotIn('RETRY', text)
            self.assertIn('request_gpus = 0', (Path(directory) / 'prepare.sub').read_text())
            for size in ('9B', '27B'):
                for c in ('C1', 'C2', 'C3'):
                    sub = (Path(directory) / f'{size}-{c}.sub').read_text()
                    self.assertIn(f'lr-rank-v2-{size}.json {c} 9 tuning', sub)
                    self.assertIn('run_uploaded_v2.sh', sub)
                    self.assertIn('GPUs_Capability == 9.0' if size == '27B' else 'GPUs_Capability == 8.0', sub)
                    self.assertNotIn('$(REGISTRY)', sub)
                    self.assertNotIn('$(MODEL_SIZE)', sub)
                    self.assertIn('$(ClusterId)', sub)


if __name__ == '__main__':
    unittest.main()
