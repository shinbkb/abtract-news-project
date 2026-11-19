# Chạy đoạn này trong cell cuối notebook Kaggle sau khi đã lưu checkpoint.
from pathlib import Path
import shutil
from IPython.display import FileLink, display
folder=Path('/kaggle/working/news_project')
for name in ['model.pt','retrieval_index.pt']:
    assert (folder/name).exists(), f'Chưa có {name}; chạy cell lưu checkpoint trước.'
archive=shutil.make_archive('/kaggle/working/news_project_output','zip',folder)
display(FileLink(archive))
