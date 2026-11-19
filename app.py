import os, time
from pathlib import Path
import pandas as pd
import streamlit as st
from src.inference import Predictor

st.set_page_config(page_title='Bản Tin · Vietnamese News NLP',page_icon='📰',layout='wide')
st.markdown("""<style>
.stApp {background: #f5f7fb;}
[data-testid="stMainBlockContainer"] {max-width:1100px;padding-top:3rem;}
h1 {letter-spacing:-.04em;} div[data-testid="stForm"] {background:white;border-radius:18px;}
</style>""",unsafe_allow_html=True)
st.caption('BẢN TIN / VIETNAMESE NEWS NLP')
st.title('Hiểu chủ đề. Khám phá bài liên quan.')
st.write('Nhập một bài báo tiếng Việt để phân loại chuyên mục và tìm tối đa ba bài có nội dung gần nhau về từ vựng.')
folder=Path(os.environ.get('ARTIFACT_DIR',str(Path(__file__).parent/'artifacts')))
@st.cache_resource
def load(path,model_mtime,index_mtime): return Predictor(path)
missing=[name for name in ['model.pt','retrieval_index.pt'] if not (folder/name).exists()]
if missing:
    st.info('Ứng dụng đang chờ mô hình đã huấn luyện. Chưa thể phân tích bài viết.')
    with st.expander('Hướng dẫn dành cho người triển khai'):
        st.write('Tải output Kaggle và nhập bằng lệnh:')
        st.code('python -m scripts.import_kaggle /đường/dẫn/news_project_output.zip')
    st.stop()
try:
    predictor=load(str(folder),(folder/'model.pt').stat().st_mtime_ns,(folder/'retrieval_index.pt').stat().st_mtime_ns)
except Exception:
    st.error('Không nạp được mô hình. Kiểm tra model và chỉ mục được xuất từ cùng một phiên huấn luyện.')
    st.stop()
with st.sidebar:
    st.subheader('Kho tin tức')
    st.metric('Chuyên mục',len(predictor.labels)); st.metric('Bài trong kho',len(predictor.retriever.titles))
    st.caption('Mô hình '+predictor.kind+' · TF-IDF tiếng Việt')
    st.divider(); st.write('Điểm softmax chưa được hiệu chuẩn. Cosine thể hiện mức tương đồng từ vựng, không phải xác suất bài liên quan.')
examples={'Tự nhập':'','Thể thao':'Đội tuyển Việt Nam chuẩn bị chiến thuật cho giải bóng đá Đông Nam Á',
          'Giáo dục':'Học sinh đăng ký xét tuyển đại học và lựa chọn ngành học',
          'Công nghệ':'Ứng dụng trí tuệ nhân tạo trong nghiên cứu và sản xuất'}
choice=st.selectbox('Bắt đầu từ một ví dụ',list(examples))
with st.form('article'):
    title=st.text_input('Tiêu đề',value=examples[choice],max_chars=500)
    content=st.text_area('Nội dung bài viết',height=200,max_chars=99000,placeholder='Dán nội dung tại đây để mô hình có thêm thông tin…')
    submitted=st.form_submit_button('Phân tích bài viết',type='primary')
if submitted:
    try:
        start=time.perf_counter(); result=predictor.predict(title,content); ms=(time.perf_counter()-start)*1000
    except ValueError as exc:
        st.warning(str(exc)); st.stop()
    st.divider(); left,right=st.columns([1,2])
    with left:
        st.caption('CHUYÊN MỤC DỰ ĐOÁN'); st.subheader(result['label']); st.caption(f'Xử lý trong {ms:.0f} ms')
    with right:
        st.bar_chart(pd.Series(result['scores']).sort_values(ascending=False).head(5))
        st.caption('5 điểm softmax cao nhất · chưa hiệu chuẩn')
    st.subheader('Có thể bạn muốn đọc')
    if not result['related']: st.info('Không tìm thấy bài có tương đồng từ vựng dương.')
    for row in result['related']:
        with st.container(border=True):
            st.caption(f"{row['category']} · Cosine {row['cosine']:.3f}")
            st.subheader(row['title']); st.write(row['excerpt'])
