import argparse, random, json
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi
from .preprocessing import normalize, fingerprint

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='configs/default.json'); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); revision=HfApi().dataset_info(cfg['dataset'],revision=cfg.get('revision')).sha
    streams=load_dataset(cfg['dataset'],revision=revision,streaming=True)
    if 'train' not in streams or 'test' not in streams: raise ValueError('Cần split train/test gốc')
    val_name=next((s for s in ['validation','val','valid','dev'] if s in streams),None)
    n=cfg['per_class']; nt=int(.7*n); nv=int(.15*n); ne=n-nt-nv
    used_content=set(); used_title=set(); logs={}
    def sample(name,quota):
        groups=defaultdict(list); stats=Counter()
        for row in streams[name].shuffle(seed=cfg['seed'],buffer_size=10000):
            stats['scanned']+=1
            if not all(isinstance(row.get(k),str) and row[k].strip() for k in ['title','content','category']): continue
            if len(row['content'].strip())<100: continue
            label=row['category'].strip()
            if len(groups[label])>=quota: continue
            h,t=fingerprint(row['content']),fingerprint(row['title'])
            if h in used_content or t in used_title: stats['duplicates']+=1; continue
            used_content.add(h); used_title.add(t)
            groups[label].append(dict(title=row['title'],content=row['content'],category=label,description=row.get('description') or '',id=row.get('id','')))
        logs[name]=dict(stats)
        return [r for group in groups.values() for r in group]
    test=sample('test',ne)
    if val_name: val=sample(val_name,nv); train=sample('train',nt)
    else:
        pool=sample('train',nt+nv); groups=defaultdict(list); train=[]; val=[]
        for row in pool: groups[row['category']].append(row)
        rng=random.Random(cfg['seed'])
        for rows in groups.values():
            rng.shuffle(rows); cut=max(1,min(nv,len(rows)//5)); val.extend(rows[:cut]); train.extend(rows[cut:])
    out=Path(cfg['data']); out.mkdir(parents=True,exist_ok=True)
    counts={}
    for name,rows in [('train',train),('val',val),('test',test)]:
        if not rows: raise ValueError(f'Split {name} rỗng')
        frame=pd.DataFrame(rows); frame.to_csv(out/f'{name}.csv',index=False); counts[name]=frame.category.value_counts().to_dict()
    (out/'dataset_summary.json').write_text(json.dumps(dict(source=cfg['dataset'],revision=revision,seed=cfg['seed'],counts=counts,logs=logs,
         validation_source=val_name or 'train',dedup='NFC lowercase whitespace content AND title; near-duplicates not guaranteed'),ensure_ascii=False,indent=2))
    from .dataset import load_splits
    load_splits(out); print(counts)
if __name__=='__main__': main()
