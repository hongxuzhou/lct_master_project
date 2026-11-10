"""Sampling invariants for paired variants and fixed condition inputs."""
from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tuning.build_subset import sample_ids
from data import condition_rows


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(example_id=f'f{i}', item_id=None, variant='pmb_fluent',
                          repair_type=None, doc_id=f'f{i}', text='fluent',
                          sbn_clean='clean', sbn_repair='clean') for i in range(100)]
        for kind in ('name', 'word'):
            for i in range(40):
                item = f'{kind}{i}'
                base = dict(item_id=item, repair_type=kind, doc_id=item,
                            text='repair', sbn_clean='clean', sbn_repair='repair')
                self.rows.append(dict(base, example_id=item+'.base', variant='rsr_base'))
                if i < 20:
                    self.rows.append(dict(base, example_id=item+'.marked', variant='rsr_editing_expression'))

    def test_proportions_pairs_and_condition_views(self):
        subset, _ = sample_ids(self.rows)
        self.assertEqual(Counter(r['variant'] for r in subset),
                         {'pmb_fluent': 10, 'rsr_base': 8, 'rsr_editing_expression': 4})
        ids = {r['example_id'] for r in subset}
        for row in self.rows:
            if row['variant'] == 'rsr_editing_expression':
                self.assertEqual(row['example_id'] in ids, row['item_id']+'.base' in ids)
        c2, c3 = condition_rows(subset, 'C2'), condition_rows(subset, 'C3')
        self.assertEqual([(r['example_id'], r['text']) for r in c2],
                         [(r['example_id'], r['text']) for r in c3])
        self.assertEqual(len(condition_rows(subset, 'C1')), 10)

    def test_membership_is_reproducible_and_source_order_is_preserved(self):
        first, _ = sample_ids(self.rows)
        second, _ = sample_ids(list(reversed(self.rows)))
        self.assertEqual({r['example_id'] for r in first}, {r['example_id'] for r in second})
        self.assertEqual(first, list(reversed(second)))
        other, _ = sample_ids(self.rows, seed=99)
        self.assertNotEqual({r['example_id'] for r in first}, {r['example_id'] for r in other})

    def test_orphan_marked_variant_is_rejected(self):
        rows = [r for r in self.rows if r['example_id'] != 'name0.base']
        with self.assertRaisesRegex(ValueError, 'Invalid repair group'):
            sample_ids(rows)

    def test_synthetic_fluent_is_in_c1_and_sampled_as_complete_units(self):
        synthetic = [dict(example_id=f'g{i}', item_id=f'g{i}', doc_id=f'g{i}',
                          variant='fluent_split_reference', text='group',
                          sbn_clean='same', sbn_repair='same', repair_type=None,
                          construction_type='same_clause', partition_unit=f'unit{i//2}')
                     for i in range(8)]
        selected, _ = sample_ids(self.rows+synthetic, fraction=.5)
        group_rows = [r for r in selected if r['variant']=='fluent_split_reference']
        self.assertEqual(len(group_rows),4)
        self.assertEqual(set(Counter(r['partition_unit'] for r in group_rows).values()),{2})
        c1 = condition_rows(selected,'C1')
        self.assertEqual(len(c1),54)
        self.assertTrue({r['example_id'] for r in group_rows}<={r['example_id'] for r in c1})
        self.assertEqual([(r['example_id'],r['text']) for r in condition_rows(selected,'C2')],
                         [(r['example_id'],r['text']) for r in condition_rows(selected,'C3')])


if __name__ == '__main__':
    unittest.main()
