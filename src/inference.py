from pathlib import Path
import torch
from .models import make_model
from .tfidf import Tfidf
from .retrieval import Retriever

class Predictor:
    def __init__(self,artifacts):
        folder=Path(artifacts)
        checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
        self.vectorizer=Tfidf.from_checkpoint(checkpoint); self.labels=checkpoint['labels']
        self.kind=checkpoint['model_type']
        self.model=make_model(self.kind,len(self.vectorizer.vocab),len(self.labels))
        self.model.load_state_dict(checkpoint['state_dict']); self.model.eval()
        self.retriever=Retriever(torch.load(folder/'retrieval_index.pt',map_location='cpu',weights_only=True))
        if self.retriever.vectors.shape[1]!=len(self.vectorizer.vocab): raise ValueError('Model và retrieval index khác vocabulary')
    @torch.inference_mode()
    def predict(self,title,content=''):
        text=(title+' '+content).strip()
        if not text: raise ValueError('Hãy nhập tiêu đề hoặc nội dung.')
        if len(text)>100000: raise ValueError('Bài viết tối đa 100.000 ký tự.')
        q=self.vectorizer.transform([text])
        if q.norm().item()==0: raise ValueError('Không có từ nằm trong vocabulary. Hãy nhập thêm nội dung tiếng Việt.')
        probs=self.model(q).softmax(1)[0]
        return dict(label=self.labels[int(probs.argmax())],scores=dict(zip(self.labels,probs.tolist())),
                    related=self.retriever.search(q,title,content))
