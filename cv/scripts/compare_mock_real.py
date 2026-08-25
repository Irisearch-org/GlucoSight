import json
from collections import Counter

real_path = "cv/features/cv/cv-baseline-v0.1/features.jsonl"
mock_path = "cv/features/cv/cv-baseline-v0.1_mock/features_mock.jsonl"

real_map = {}
with open(real_path, encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        real_map[rec['sample_id']] = rec

mock_map = {}
with open(mock_path, encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        mock_map[rec['sample_id']] = rec

common = set(real_map) & set(mock_map)
only_real = set(real_map) - set(mock_map)
only_mock = set(mock_map) - set(real_map)

match_top1 = 0
match_top1_and_ok = 0
conf_diffs = []

for sid in sorted(common):
    r = real_map[sid]
    m = mock_map[sid]
    if r.get('food_top1') == m.get('food_top1'):
        match_top1 += 1
        if r.get('feature_status') == 'ok':
            match_top1_and_ok += 1
    rc = r.get('confidence') or 0
    mc = m.get('confidence') or 0
    conf_diffs.append(abs(rc - mc))

import statistics
out = {
    'total_real': len(real_map),
    'total_mock': len(mock_map),
    'common': len(common),
    'only_real': len(only_real),
    'only_mock': len(only_mock),
    'top1_match_count': match_top1,
    'top1_match_pct': round(100 * match_top1 / len(common), 2) if common else None,
    'top1_match_and_real_ok_count': match_top1_and_ok,
    'mean_conf_diff': round(statistics.mean(conf_diffs),4) if conf_diffs else None,
    'median_conf_diff': round(statistics.median(conf_diffs),4) if conf_diffs else None,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
