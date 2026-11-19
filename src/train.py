import argparse, copy, json, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from .dataset import load_splits
from .tfidf import Tfidf
from .models import make_model
from .evaluate import evaluate
from .preprocessing import fingerprint

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='configs/default.json'); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); out=Path(cfg['artifacts']); out.mkdir(parents=True,exist_ok=True)
    parts=load_splits(cfg['data']); labels=sorted(parts['train'].category.unique())
    tf=Tfidf(cfg['max_features'],cfg['min_df']).fit(parts['train'].text)
    x={n:tf.transform(parts[n].text) for n in ['train','val']}
    y={n:torch.tensor([labels.index(v) for v in parts[n].category]) for n in x}
    device='cuda' if torch.cuda.is_available() else 'cpu'; comparison=[]; histories={}
    for kind,lr in [('Linear',cfg['linear_lr']),('MLP',cfg['mlp_lr'])]:
        random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        model=make_model(kind,len(tf.vocab),len(labels)).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        loader=DataLoader(TensorDataset(x['train'],y['train']),batch_size=cfg['batch_size'],shuffle=True)
        best=-1; stale=0; history=[]
        for epoch in range(1,cfg['epochs']+1):
            model.train(); loss_sum=0
            for xb,yb in loader:
                optimizer.zero_grad(); loss=torch.nn.functional.cross_entropy(model(xb.to(device)),yb.to(device))
                loss.backward(); optimizer.step(); loss_sum+=loss.item()*len(yb)
            metrics,_=evaluate(model,x['val'],y['val'],len(labels),device)
            history.append(dict(epoch=epoch,train_loss=loss_sum/len(y['train']),val_macro_f1=metrics['macro_f1']))
            print(kind,history[-1],flush=True)
            if metrics['macro_f1']>best+1e-5:
                best=metrics['macro_f1']; stale=0; best_epoch=epoch
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                stale+=1
                if stale>=cfg['patience']: break
        checkpoint=dict(model_type=kind,state_dict=state,vocab=tf.vocab,idf=tf.idf,labels=labels,max_features=cfg['max_features'])
        torch.save(checkpoint,out/f'best_{kind.lower()}.pt'); histories[kind]=history
        comparison.append(dict(model=kind,val_macro_f1=best,best_epoch=best_epoch))
    chosen=max(comparison,key=lambda row:row['val_macro_f1'])['model']
    import shutil
    shutil.copyfile(out/f'best_{chosen.lower()}.pt',out/'model.pt')
    train=parts['train']
    torch.save(dict(vectors=x['train'],titles=train.title.tolist(),categories=train.category.tolist(),excerpts=train.content.str[:250].tolist(),
                    content_hashes=train.content.map(fingerprint).tolist(),text_hashes=train.text.map(fingerprint).tolist()),out/'retrieval_index.pt')
    (out/'vocabulary.json').write_text(json.dumps(tf.vocab,ensure_ascii=False),encoding='utf-8'); torch.save(tf.idf,out/'idf.pt')
    (out/'labels.json').write_text(json.dumps(labels,ensure_ascii=False),encoding='utf-8')
    (out/'validation_comparison.json').write_text(json.dumps(comparison,indent=2))
    (out/'history.json').write_text(json.dumps(histories,indent=2))
    print('Selected:',chosen,'— test chưa được đánh giá. Chạy python -m src.evaluate khi đã chốt.')
if __name__=='__main__': main()
