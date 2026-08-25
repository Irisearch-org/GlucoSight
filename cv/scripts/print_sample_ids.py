import json
f='cv/features/cv/cv-baseline-v0.1/features.jsonl'
for i,line in enumerate(open(f,encoding='utf-8')):
    if i>=20: break
    r=json.loads(line)
    print(i, repr(r.get('sample_id')))
