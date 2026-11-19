import argparse, json
from pathlib import Path
import torch
import pandas as pd
from .inference import Predictor
from .dataset import load_splits
from .metrics import metrics_from_predictions

@torch.inference_mode()
def evaluate(model,x,y,classes,device='cpu'):
    model.eval()
    pred=torch.cat([model(b.to(device)).argmax(1).cpu() for b in x.split(256)])
    return metrics_from_predictions(y,pred,classes),pred

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data',default='data'); parser.add_argument('--artifacts',default='artifacts'); args=parser.parse_args()
    p=Predictor(args.artifacts); df=load_splits(args.data)['test']
    x=p.vectorizer.transform(df.text); y=torch.tensor([p.labels.index(v) for v in df.category])
    scores,pred=evaluate(p.model,x,y,len(p.labels)); cm=scores.pop('confusion_matrix')
    output=Path(args.artifacts)/'evaluation'; output.mkdir(exist_ok=True)
    (output/'test_metrics.json').write_text(json.dumps(scores,indent=2))
    tp=cm.diag().float(); precision=tp/cm.sum(0).clamp_min(1); recall=tp/cm.sum(1).clamp_min(1)
    f1=2*precision*recall/(precision+recall).clamp_min(1e-12)
    pd.DataFrame(dict(label=p.labels,precision=precision.tolist(),recall=recall.tolist(),f1=f1.tolist(),support=cm.sum(1).tolist())).to_csv(output/'per_class.csv',index=False)
    pd.DataFrame(cm.tolist(),index=p.labels,columns=p.labels).to_csv(output/'confusion_matrix.csv')
    df['prediction']=[p.labels[i] for i in pred.tolist()]; df[df.category!=df.prediction].to_csv(output/'errors.csv',index=False)
    print(scores)
if __name__=='__main__': main()
