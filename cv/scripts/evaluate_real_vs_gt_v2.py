import json
import os
from collections import Counter

features_path = "cv/features/cv/cv-baseline-v0.1/features.jsonl"
dataset_test_dir = "data/cv/indonesian_food_image/Indonesian Food Image/Clean_Data/test"

# Load feature records
records = []
with open(features_path, encoding='utf-8') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

# Totals and ok count
total = len(records)
ok_count = sum(1 for r in records if r.get('feature_status') == 'ok')

# Build ground-truth map: map basename without extension -> class label
gt_map = {}
for cls in os.listdir(dataset_test_dir):
    cls_dir = os.path.join(dataset_test_dir, cls)
    if not os.path.isdir(cls_dir):
        continue
    for fname in os.listdir(cls_dir):
        name_no_ext = os.path.splitext(fname)[0]
        gt_map[name_no_ext] = cls

# Evaluate top-1 accuracy
correct = 0
missing_gt = 0
per_class = Counter()
per_class_correct = Counter()

for r in records:
    sid = r.get('sample_id')
    top1 = r.get('food_top1')
    # normalize sample id by stripping path and extension
    bn = os.path.splitext(os.path.basename(sid))[0]
    gt = gt_map.get(bn)
    if gt is None:
        missing_gt += 1
        continue
    per_class[gt] += 1
    if top1 == gt:
        correct += 1
        per_class_correct[gt] += 1

accuracy = correct / (total - missing_gt) if (total - missing_gt) > 0 else None

import statistics
out = {
    'total_samples_in_features': total,
    'ok_count': ok_count,
    'ok_pct': round(100 * ok_count / total, 2) if total else None,
    'missing_ground_truth_count': missing_gt,
    'evaluated_samples': total - missing_gt,
    'top1_correct': correct,
    'top1_accuracy_pct': round(100 * accuracy, 2) if accuracy is not None else None,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
