import json
from collections import Counter, defaultdict
import statistics

path = "cv/features/cv/cv-baseline-v0.1/features.jsonl"
counts = Counter()
confidences = []
low_conf = 0
per_class_conf = defaultdict(list)
total = 0

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        total += 1
        top1 = rec.get("food_top1")
        counts[top1] += 1
        c = rec.get("confidence")
        if isinstance(c, (int, float)):
            confidences.append(c)
            per_class_conf[top1].append(c)
        if rec.get("feature_status") != "ok":
            low_conf += 1

# stats
mean_conf = statistics.mean(confidences) if confidences else 0
median_conf = statistics.median(confidences) if confidences else 0
stdev_conf = statistics.pstdev(confidences) if confidences else 0

# top classes
top_classes = counts.most_common(10)
per_class_mean = {k: (statistics.mean(v) if v else 0) for k,v in per_class_conf.items()}

import json
out = {
    "total": total,
    "low_confidence_count": low_conf,
    "low_confidence_pct": round(100*low_conf/total,2) if total else 0,
    "mean_confidence": round(mean_conf,4),
    "median_confidence": round(median_conf,4),
    "stdev_confidence": round(stdev_conf,4),
    "top_classes": top_classes,
    "per_class_mean_confidence_top10": {k: round(per_class_mean[k],4) for k,_ in top_classes}
}
print(json.dumps(out, ensure_ascii=False))
