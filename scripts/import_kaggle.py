"""Import trusted Kaggle artifacts without executing pickle or extracting arbitrary paths."""
import argparse, json, shutil, tempfile, zipfile
from pathlib import Path
import torch
from src.inference import Predictor
from src.preprocessing import fingerprint

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('source'); parser.add_argument('--replace',action='store_true'); args=parser.parse_args()
    source=Path(args.source).expanduser().resolve()
    names=['model.pt','retrieval_index.pt','train.csv','val.csv','test.csv','labels.json']
    if not args.replace and any((ROOT/'artifacts'/n).exists() for n in names[:2]):
        raise SystemExit('Artifacts đã tồn tại. Dùng --replace nếu muốn thay model hiện tại.')
    with tempfile.TemporaryDirectory() as tmp:
        stage=Path(tmp)
        if source.is_dir():
            for name in names:
                matches=list(source.rglob(name))
                if len(matches)>1: raise ValueError(f'Nhiều file {name}; hãy chọn thư mục cụ thể')
                if matches: shutil.copyfile(matches[0],stage/name)
        else:
            with zipfile.ZipFile(source) as archive:
                if sum(i.file_size for i in archive.infolist())>2_000_000_000: raise ValueError('ZIP vượt 2GB')
                for name in names:
                    matches=[i for i in archive.infolist() if not i.is_dir() and Path(i.filename).name==name]
                    if len(matches)>1: raise ValueError(f'Nhiều file {name} trong ZIP')
                    if matches:
                        with archive.open(matches[0]) as src, (stage/name).open('wb') as dst: shutil.copyfileobj(src,dst)
        for name in names[:2]:
            if not (stage/name).exists(): raise FileNotFoundError(f'Thiếu {name}')
        predictor=Predictor(stage)
        validation='Shape checked only; train.csv missing, index correspondence not verified'
        if (stage/'train.csv').exists():
            import pandas as pd
            frame=pd.read_csv(stage/'train.csv',keep_default_na=False)
            index=torch.load(stage/'retrieval_index.pt',map_location='cpu',weights_only=True)
            if frame.title.tolist()!=index['titles'] or frame.category.tolist()!=index['categories']:
                raise ValueError('train.csv và retrieval index không cùng thứ tự/bộ dữ liệu')
            # Verify every row, in bounded batches, against the checkpoint vocabulary/IDF.
            for start in range(0,len(frame),128):
                batch=frame.iloc[start:start+128]
                actual=predictor.vectorizer.transform(batch.title+' '+batch.content)
                if not torch.allclose(actual,index['vectors'][start:start+128].cpu(),atol=1e-5,rtol=1e-4):
                    raise ValueError('Vector index không khớp model và train.csv')
            index['content_hashes']=frame.content.map(fingerprint).tolist()
            index['text_hashes']=(frame.title+' '+frame.content).map(fingerprint).tolist()
            torch.save(index,stage/'retrieval_index.pt')
            validation='All retrieval rows verified against checkpoint and train.csv'
        for name in names:
            if (stage/name).exists():
                dest=ROOT/('data' if name.endswith('.csv') else 'artifacts')/name
                dest.parent.mkdir(exist_ok=True); shutil.copyfile(stage/name,dest)
        a=ROOT/'artifacts'; tf=predictor.vectorizer
        (a/'vocabulary.json').write_text(json.dumps(tf.vocab,ensure_ascii=False),encoding='utf-8')
        (a/'labels.json').write_text(json.dumps(predictor.labels,ensure_ascii=False),encoding='utf-8'); torch.save(tf.idf,a/'idf.pt')
        (a/'import_report.json').write_text(json.dumps(dict(model=predictor.kind,labels=len(predictor.labels),documents=len(predictor.retriever.titles),validation=validation),indent=2))
        print('Imported:',predictor.kind,validation)
if __name__=='__main__': main()
