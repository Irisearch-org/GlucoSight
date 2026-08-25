import json
import os
from collections import Counter

# Paths
features_path = "cv/features/cv/cv-baseline-v0.1/features.jsonl"
dataset_test_dir = "data/cv/indonesian_food_image/Indonesian Food Image/Clean_Data/test"

# Load features
records = []
with open(features_path, encoding='utf-8') as f:
    for line in f:
        records.append(json.loads(line))

# Compute ok counts
total = len(records)
ok_count = sum(1 for r in records if r.get('feature_status') == 'ok')

# Build ground-truth map from test folder structure
# Assuming sample_id == relative path like "test/classname/filename.jpg" or just filename
# We'll map by filename fallback
gt_map = {}
for cls in os.listdir(dataset_test_dir):
    cls_dir = os.path.join(dataset_test_dir, cls)
    if not os.path.isdir(cls_dir):
        continue
    for fname in os.listdir(cls_dir):
        gt_map[fname] = cls
        # also include full relative path keys
        gt_map[os.path.join(cls, fname)] = cls

# Evaluate top-1 accuracy
correct = 0
missing_gt = 0
per_class = Counter()
per_class_correct = Counter()

for r in records:
    sid = r.get('sample_id')
    top1 = r.get('food_top1')
    # Try to find ground truth
    gt = None
    if sid in gt_map:
        gt = gt_map[sid]
    else:
        # try basename
        import ntpath
        bn = ntpath.basename(sid)
        if bn in gt_map:
            gt = gt_map[bn]
    if gt is None:
        missing_gt += 1
        continue
    per_class[gt] += 1
    if top1 == gt:
        correct += 1
        per_class_correct[gt] += 1

# Output
import math
accuracy = correct / (total - missing_gt) if (total - missing_gt) > 0 else None
out = {
    'total_samples': total,
    'ok_count': ok_count,
    'ok_pct': round(100 * ok_count / total, 2) if total else None,
    'missing_ground_truth': missing_gt,
    'evaluated_samples': total - missing_gt,
    'top1_correct': correct,
    'top1_accuracy_pct': round(100 * accuracy, 2) if accuracy is not None else None,
    'per_class_counts': dict(per_class),
    'per_class_correct': dict(per_class_correct)
}
print(json.dumps(out, ensure_ascii=False, indent=2))
