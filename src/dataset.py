from pathlib import Path
import pandas as pd
from .preprocessing import fingerprint

def load_splits(directory):
    parts={n:pd.read_csv(Path(directory)/f'{n}.csv',keep_default_na=False) for n in ['train','val','test']}
    for name,df in parts.items():
        if not {'title','content','category'}.issubset(df.columns) or df.empty: raise ValueError(f'Split không hợp lệ: {name}')
        df['text']=df.title+' '+df.content
    labels=set(parts['train'].category)
    if any(set(df.category)!=labels for df in parts.values()): raise ValueError('Nhãn giữa các split không khớp')
    for field in ['content','title']:
        hashes={n:set(df[field].map(fingerprint)) for n,df in parts.items()}
        for a,b in [('train','val'),('train','test'),('val','test')]:
            overlap=hashes[a]&hashes[b]
            if overlap: raise ValueError(f'Phát hiện {len(overlap)} {field} trùng giữa {a}/{b}; cần chia lại và train lại.')
    return parts
