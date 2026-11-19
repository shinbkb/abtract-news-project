"""Prepare a minimal deploy directory; no credentials, test data, notebook outputs."""
from pathlib import Path
import argparse, shutil
from src.inference import Predictor
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(); p.add_argument('destination'); args=p.parse_args()
    Predictor(ROOT/'artifacts')
    dest=Path(args.destination).resolve()
    if dest.exists(): raise SystemExit('Destination phải chưa tồn tại.')
    dest.mkdir(parents=True)
    for name in ['app.py','requirements.txt','Dockerfile']:
        shutil.copyfile(ROOT/name,dest/name)
    for name in ['src','.streamlit']:
        shutil.copytree(ROOT/name,dest/name,ignore=shutil.ignore_patterns('__pycache__','secrets.toml'))
    (dest/'artifacts').mkdir()
    for name in ['model.pt','retrieval_index.pt']:
        shutil.copyfile(ROOT/'artifacts'/name,dest/'artifacts'/name)
    shutil.copyfile(ROOT/'deploy/SPACE_README.md',dest/'README.md')
    print('Ready:',dest)
if __name__=='__main__': main()
