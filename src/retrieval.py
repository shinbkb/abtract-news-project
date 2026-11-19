import torch
from .preprocessing import normalize

class Retriever:
    def __init__(self,index):
        self.vectors=index['vectors'].float().cpu()
        self.titles=index['titles']; self.categories=index['categories']; self.excerpts=index['excerpts']
        self.content_hashes=index.get('content_hashes')
        self.text_hashes=index.get('text_hashes')
        for hashes in [self.content_hashes,self.text_hashes]:
            if hashes is not None and len(hashes)!=len(self.titles): raise ValueError('Hash index không khớp metadata')
        if self.vectors.ndim!=2 or any(len(v)!=len(self.vectors) for v in [self.titles,self.categories,self.excerpts]):
            raise ValueError('Index không khớp metadata')
        if not torch.isfinite(self.vectors).all(): raise ValueError('Index chứa giá trị không hợp lệ')
        self.vectors=torch.nn.functional.normalize(self.vectors,p=2,dim=1)
    def search(self,q,title='',content='',k=3):
        from .preprocessing import fingerprint
        scores=(q@self.vectors.T).squeeze(0)
        for i,t in enumerate(self.titles):
            duplicate=bool(title.strip()) and normalize(t)==normalize(title)
            if content.strip() and self.content_hashes: duplicate |= fingerprint(content)==self.content_hashes[i]
            if self.text_hashes: duplicate |= fingerprint(title+' '+content)==self.text_hashes[i]
            if duplicate: scores[i]=-1
        count=min(max(0,k),int((scores>0).sum()))
        if not count: return []
        values,ids=scores.topk(count)
        return [dict(title=self.titles[i],category=self.categories[i],excerpt=self.excerpts[i],cosine=float(v))
                for v,i in zip(values.tolist(),ids.tolist())]
