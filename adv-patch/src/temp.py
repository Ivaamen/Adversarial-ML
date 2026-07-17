import csv

rows = list(csv.DictReader(open('outputREAL/train_log.csv')))

def bucket(cr):
    if cr < 0.1: return '<0.1'
    if cr < 0.25: return '0.1-0.25'
    if cr < 0.5: return '0.25-0.5'
    if cr < 1.0: return '0.5-1.0'
    return '>=1.0'

from collections import defaultdict
groups = defaultdict(list)
for r in rows:
    groups[bucket(float(r['coverage_ratio']))].append(float(r['obj_loss']))

for k in ['<0.1','0.1-0.25','0.25-0.5','0.5-1.0','>=1.0']:
    v = groups.get(k, [])
    if v:
        print(f'{k}: n={len(v)} mean_obj_loss={sum(v)/len(v):.4f}')