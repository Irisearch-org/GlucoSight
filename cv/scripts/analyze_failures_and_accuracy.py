import json
import os
from collections import Counter, defaultdict

features_path = "cv/features/cv/cv-baseline-v0.1/features.jsonl"
dataset_test_dir = "data/cv/indonesian_food_image/Indonesian Food Image/Clean_Data/test"

records = []
with open(features_path, encoding='utf-8') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

total = len(records)
status_counts = Counter(r.get('feature_status', 'missing') for r in records)

# Collect failure reasons
reason_counter = Counter()
reason_examples = defaultdict(list)
for r in records:
    status = r.get('feature_status', 'missing')
    if status == 'ok':
        continue
    # Attempt to extract a human-readable reason from common fields
    reason = None
    for key in ['error', 'error_message', 'notes', 'reason', 'feature_status_reason', 'status_message']:
        if key in r and r.get(key):
            reason = r.get(key)
            break
    # Fallback: if confidence low, mark as low_confidence
    if not reason:
        if status == 'low_confidence':
            reason = 'low_confidence'
        else:
            reason = 'unknown'
    reason_str = str(reason)
    reason_counter[reason_str] += 1
    if len(reason_examples[reason_str]) < 10:
        reason_examples[reason_str].append(r.get('sample_id'))

# Build ground-truth map by basename -> class
gt_map = {}
for cls in os.listdir(dataset_test_dir):
    cls_dir = os.path.join(dataset_test_dir, cls)
    if not os.path.isdir(cls_dir):
        continue
    for fname in os.listdir(cls_dir):
        name_no_ext = os.path.splitext(fname)[0]
        gt_map[name_no_ext] = cls

# Accuracy overall and among ok samples
def basename_noext(sid):
    return os.path.splitext(os.path.basename(sid))[0]

correct_all = 0
missing_gt = 0
correct_ok = 0
count_ok = 0

for r in records:
    sid = r.get('sample_id')
    bn = basename_noext(sid)
    gt = gt_map.get(bn)
    if gt is None:
        missing_gt += 1
        continue
    top1 = r.get('food_top1')
    if top1 == gt:
        correct_all += 1
    if r.get('feature_status') == 'ok':
        count_ok += 1
        if top1 == gt:
            correct_ok += 1

evaluated = total - missing_gt
accuracy_all = correct_all / evaluated if evaluated else None
accuracy_ok = correct_ok / count_ok if count_ok else None

out = {
    'total_samples_in_features': total,
    'status_counts': dict(status_counts),
    'failure_reason_counts_top': reason_counter.most_common(),
    'failure_reason_examples': {k: v for k, v in reason_examples.items()},
    'missing_ground_truth_count': missing_gt,
    'evaluated_samples': evaluated,
    'top1_correct_all': correct_all,
    'top1_accuracy_all_pct': round(100 * accuracy_all, 2) if accuracy_all is not None else None,
    'ok_samples_count': count_ok,
    'top1_correct_ok': correct_ok,
    'top1_accuracy_ok_pct': round(100 * accuracy_ok, 2) if accuracy_ok is not None else None,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
