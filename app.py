# app.py
import streamlit as st
from preprocessing import preprocessing
from modules import run_module1, run_module2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Đọc dữ liệu từ CSV
career_link = pd.read_csv(r'C:\Users\trand\Documents\study\university\23_24_ki1\streamlit_career\MODEL\career_link_processed_content.csv')
career_link['Content'] = career_link['Content'].apply(eval)

# Hàm tiền xử lý
def process_resume_and_recommend(resume_text):
    # Các bước tiền xử lý dữ liệu người dùng
    resume_text = preprocessing(resume_text)
    resume_doc = ' '.join(resume_text)

    # Tính toán cosine similarities và lấy top 30 công việc tương đồng
    resume_vector = app_tfidf_vectorizer.transform([resume_doc])
    cosine_similarities = cosine_similarity(resume_vector, app_tfidf_matrix)
    top_30_job_indices = cosine_similarities.argsort(axis=1)[:, -30:][0][::-1]
    top_30_job_titles = [career_link.loc[i, 'Job Title'] for i in top_30_job_indices]
    top_30_job_info = [career_link.loc[i] for i in top_30_job_indices]

    # Tạo DataFrame với thông tin top 30 công việc
    top_30_job_df = pd.DataFrame(top_30_job_info)
    top_30_job_df['Job Title'] = top_30_job_titles

    # Giữ lại cột "Content" trong top_30_job_df
    top_30_job_df['Content'] = career_link.loc[top_30_job_indices, 'Content'].tolist()

    # Loại bỏ cột "Content" nếu bạn không muốn giữ lại nó trong top_30_job_df
    top_30_job_df = top_30_job_df.drop(columns=['Content'], axis=1)

    return top_30_job_df
    



# Tạo TF-IDF Vectorizer và ma trận
app_documents = [' '.join(tokens) for tokens in career_link['Content']]
app_tfidf_vectorizer = TfidfVectorizer()
app_tfidf_matrix = app_tfidf_vectorizer.fit_transform(app_documents)

# Thiết lập web app
def main():
    st.title("Ứng dụng Streamlit với 2 lựa chọn")

    option1 = st.checkbox("theo hồ sơ")
    option2 = st.checkbox("theo từ khóa")

    if option1:
        run_module1(process_resume_and_recommend)

    if option2:
        run_module2(career_link, app_tfidf_matrix)

if __name__ == "__main__":
    main()
