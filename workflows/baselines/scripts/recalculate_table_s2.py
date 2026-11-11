#!/usr/bin/env python3
"""Recalculate Supplementary Table S2 from archived Exact88 run records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

METHOD_RUNS = {
    'MolToken-Editor': ('deterministic',),
    'MMP-Transformer': ('20260703', '20260704', '20260705'),
    'REINVENT4': ('20260703', '20260704', '20260705'),
}
METRICS = (
    'Total generation attempts',
    'Model–molecule input records',
    'Structural guardrail failure rate (%)',
    'Candidates with starting-molecule Tanimoto similarity < 0.35 (%)',
)
# Display values in the submitted table, including its rounding.
PUBLISHED = (
    ('4,360', '8,496.67 ± 29.26', '4,400'),
    ('177,990', '107,960 ± 563', '76,513 ± 6,289'),
    ('44.09', '17.58 ± 3.10', '15.33 ± 0.66'),
    ('30.91', '3.11 ± 0.57', '0.45 ± 0.45'),
)
UPPER_LIMITS = {
    'edited_sa_score': 6,
    'edited_mw': 650,
    'edited_tpsa': 160,
    'edited_hbd': 5,
    'edited_hba': 12,
    'edited_rotbonds': 12,
}
COUNT_FIELDS = (
    'generation_attempts',
    'localization_model_molecule_records',
    'native_oracle_model_molecule_records',
    'candidate_scoring_model_molecule_records',
)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def flag(row, key):
    value = row[key].lower()
    check(value in ('true', 'false', '0', '1'), f'Invalid {key}: {value}')
    return value in ('true', '1')


def numeric(row, key):
    value = float(row[key])
    check(math.isfinite(value), f'Nonfinite {key}')
    return value


def guardrail_checks(row):
    parent = numeric(row, 'parent_tanimoto')
    train = numeric(row, 'nearest_train_tanimoto_edited')
    parent_failed = parent < 0.35
    other_failed = (
        train < 0.30 or train > 0.95 or numeric(row, 'edited_qed') < 0.35
        or flag(row, 'edited_interference_alert') or not flag(row, 'edited_valid')
        or any(numeric(row, key) > limit for key, limit in UPPER_LIMITS.items())
    )
    failed = parent_failed or other_failed
    check(failed == (not flag(row, 'relaxed_guardrail_pass')),
          f"Guardrail discrepancy for {row['method']} {row['run_id']} "
          f"{row['mol_id']} slot {row['pre_score_rank_in_parent']}")
    return failed, parent_failed, parent_failed and not other_failed


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def recalculate(data_dir, repo):
    provenance = json.loads((data_dir / 'provenance.json').read_text(encoding='utf-8'))
    names = ('run_costs.csv', 'locked_candidate_guardrails.csv')
    for name in names:
        digest = hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        check(digest == provenance['released_file_sha256'][name], f'Input hash differs: {name}')
    parents = set(provenance['starting_molecule_ids'])
    check(len(parents) == 88, 'Expected 88 unique starting-molecule IDs')
    roster = repo / 'workflows/baselines/inputs/common_exact88/phase26_locked_exact88.csv'
    if roster.is_file():
        check({row['parent_id'] for row in read_csv(roster)} == parents,
              'Repository Exact88 roster differs from Table S2')

    costs = {}
    for row in read_csv(data_dir / names[0]):
        key = row['method'], row['run_id']
        check(key not in costs, f'Duplicate run costs for {key}')
        values = [int(row[field]) for field in COUNT_FIELDS]
        check(min(values) >= 0, f'Negative counts in {key}')
        costs[key] = values[0], sum(values[1:])

    candidates = defaultdict(dict)
    for row in read_csv(data_dir / names[1]):
        key = row['method'], row['run_id']
        parent, slot = row['mol_id'], int(row['pre_score_rank_in_parent'])
        check(parent in parents and 1 <= slot <= 5 and (parent, slot) not in candidates[key],
              f'Unexpected or duplicate candidate slot in {key}: {parent}, {slot}')
        candidates[key][parent, slot] = guardrail_checks(row)

    expected = {(method, run) for method, runs in METHOD_RUNS.items() for run in runs}
    check(costs.keys() == candidates.keys() == expected, 'Unexpected run identities')
    values = defaultdict(list)
    for method, runs in METHOD_RUNS.items():
        for run in runs:
            key = method, run
            slots = candidates[key]
            per_parent = Counter(parent for parent, _ in slots)
            check(per_parent.keys() == parents, f'Starting-molecule roster differs in {key}')
            filled = len(slots)
            check(filled == (438 if key == ('REINVENT4', '20260705') else 440),
                  f'Unexpected filled-slot count in {key}: {filled}')
            failed, parent_failed, only_parent = map(sum, zip(*slots.values()))
            if method == 'MolToken-Editor':
                check(only_parent == 88, 'Table S2 note differs')
            attempts, model_records = costs[key]
            values[method].append((attempts, model_records, 100 * failed / filled,
                                   100 * parent_failed / filled))
    return values


def display(values, digits, show_zero_sd=True):
    if len(values) == 1 or (not show_zero_sd and len(set(values)) == 1):
        digits = 0 if digits == 2 and isinstance(values[0], int) else digits
        return format(values[0], f',.{digits}f')
    return f'{mean(values):,.{digits}f} ± {stdev(values):,.{digits}f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--data-dir', type=Path, help='Override the Table S2 evidence directory')
    args = parser.parse_args()
    values = recalculate(args.data_dir or args.repo / 'results/table_s2', args.repo)
    rows = tuple(
        tuple(display([run[index] for run in values[method]], digits, show_zero_sd=index != 0)
              for method in METHOD_RUNS)
        for index, digits in enumerate((2, 0, 2, 2))
    )
    check(rows == PUBLISHED, f'Recalculated table differs from the manuscript: {rows}')
    print('TABLE_S2_RECALCULATION_PASS')
    print('Metric | ' + ' | '.join(METHOD_RUNS))
    for metric, row in zip(METRICS, rows):
        print(metric + ' | ' + ' | '.join(row))
    print('MolToken-Editor candidates failing only the starting-molecule similarity check: 88/440')


if __name__ == '__main__':
    main()
