import tempfile, unittest
from pathlib import Path
import torch
from src.tfidf import Tfidf
from src.models import make_model
from src.inference import Predictor
from src.metrics import metrics_from_predictions
from src.retrieval import Retriever
from src.preprocessing import fingerprint

class PipelineTests(unittest.TestCase):
    def test_tfidf_train_only(self):
        tf=Tfidf(min_df=1).fit(['mèo mèo chó','chó chim'])
        self.assertAlmostEqual(float(tf.idf[tf.vocab['chó']]),1)
        expected=torch.zeros(len(tf.vocab)); expected[tf.vocab['mèo']]=2*tf.idf[tf.vocab['mèo']]; expected[tf.vocab['chó']]=1
        self.assertTrue(torch.allclose(tf.transform(['mèo mèo chó'])[0],expected/expected.norm()))
        before=dict(tf.vocab); self.assertEqual(tf.transform(['zzunknown']).norm().item(),0); self.assertEqual(tf.vocab,before)
    def test_metrics(self):
        m=metrics_from_predictions(torch.tensor([0,0,1,1]),torch.tensor([0,1,1,1]),2)
        self.assertEqual(m['confusion_matrix'].tolist(),[[1,1],[0,2]])
        self.assertAlmostEqual(m['macro_f1'],(2/3+4/5)/2,places=6)
    def test_reload_and_exclude_source(self):
        titles=['Bóng đá','Học sinh']; contents=['cầu thủ ghi bàn','đại học thi tuyển']
        tf=Tfidf(min_df=1).fit([a+' '+b for a,b in zip(titles,contents)])
        for kind in ['Linear','MLP']:
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp); model=make_model(kind,len(tf.vocab),2); model.eval()
                torch.save(dict(model_type=kind,state_dict=model.state_dict(),vocab=tf.vocab,idf=tf.idf,labels=['Thể thao','Giáo dục']),path/'model.pt')
                torch.save(dict(vectors=tf.transform([a+' '+b for a,b in zip(titles,contents)]),titles=titles,categories=['Thể thao','Giáo dục'],excerpts=contents,
                       content_hashes=[fingerprint(c) for c in contents]),path/'retrieval_index.pt')
                p=Predictor(path); q=tf.transform(['cầu thủ'])
                with torch.no_grad(): self.assertTrue(torch.equal(model(q),p.model(q)))
                self.assertEqual(p.predict(titles[0],contents[0])['related'],[])
                self.assertEqual(p.predict('cầu thủ')['related'][0]['title'],titles[0])
                with self.assertRaises(ValueError): p.predict('zzunknown')
                with self.assertRaises(ValueError): p.predict('')

class WebTests(unittest.TestCase):
    def test_missing_artifacts_screen(self):
        import os
        from unittest.mock import patch
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'ARTIFACT_DIR':tmp}):
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=20)
            self.assertEqual(len(app.exception),0)
            self.assertIn('chờ mô hình',app.info[0].value)
    def test_submit_with_fixture_model(self):
        import os
        from unittest.mock import patch
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); tf=Tfidf(min_df=1).fit(['bóng đá cầu thủ','đại học học sinh'])
            model=make_model('MLP',len(tf.vocab),2)
            torch.save(dict(model_type='MLP',state_dict=model.state_dict(),vocab=tf.vocab,idf=tf.idf,labels=['Thể thao','Giáo dục']),path/'model.pt')
            torch.save(dict(vectors=tf.transform(['bóng đá cầu thủ','đại học học sinh']),titles=['Bóng đá','Đại học'],categories=['Thể thao','Giáo dục'],excerpts=['Cầu thủ thi đấu','Học sinh thi tuyển']),path/'retrieval_index.pt')
            with patch.dict(os.environ,{'ARTIFACT_DIR':tmp}):
                app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=20)
                app.text_input[0].set_value('cầu thủ')
                app.button[0].click().run()
                self.assertEqual(len(app.exception),0)
                self.assertTrue(any('Có thể bạn muốn đọc' in x.value for x in app.subheader), str(app))

if __name__=='__main__': unittest.main()
