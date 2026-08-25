import json
from collections import Counter
import sys

# Simple comparator for two feature files keyed by sample_id

def load_map(path):
    m = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            m[rec['sample_id']] = rec
    return m

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: compare_models.py <fileA.jsonl> <fileB.jsonl>')
        sys.exit(1)
    a = load_map(sys.argv[1])
    b = load_map(sys.argv[2])
    common = set(a) & set(b)
    total = len(common)
    if total == 0:
        print('No common samples')
        sys.exit(1)
    top1_same = sum(1 for s in common if a[s].get('food_top1') == b[s].get('food_top1'))
    print(json.dumps({
        'samples_compared': total,
        'top1_same': top1_same,
        'top1_same_pct': round(100*top1_same/total,2)
    }, indent=2))
