import math
from collections import Counter
import torch
from .preprocessing import tokenize

class Tfidf:
    def __init__(self, max_features=4000, min_df=2):
        self.max_features, self.min_df = max_features, min_df
    def fit(self,texts):
        docs=[tokenize(t) for t in texts]; df=Counter()
        for tokens in docs: df.update(set(tokens))
        words=[w for w,n in sorted(df.items(),key=lambda kv:(-kv[1],kv[0])) if n>=self.min_df][:self.max_features]
        if not words: raise ValueError('Vocabulary rỗng')
        self.vocab={w:i for i,w in enumerate(words)}
        self.idf=torch.tensor([math.log((1+len(docs))/(1+df[w]))+1 for w in words],dtype=torch.float32)
        return self
    def transform(self,texts):
        texts=list(texts); x=torch.zeros((len(texts),len(self.vocab)),dtype=torch.float32)
        for i,text in enumerate(texts):
            counts=Counter(t for t in tokenize(text) if t in self.vocab); total=sum(counts.values())
            if total:
                for word,count in counts.items(): x[i,self.vocab[word]]=count/total
        x*=self.idf.unsqueeze(0)
        return torch.nn.functional.normalize(x,p=2,dim=1)
    @classmethod
    def from_checkpoint(cls,checkpoint):
        obj=cls(); obj.vocab=checkpoint['vocab']; obj.idf=checkpoint['idf'].float().cpu()
        if sorted(obj.vocab.values())!=list(range(len(obj.vocab))): raise ValueError('Vocabulary index không liên tục')
        if obj.idf.shape!=(len(obj.vocab),) or not torch.isfinite(obj.idf).all(): raise ValueError('IDF không hợp lệ')
        return obj
