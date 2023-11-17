import streamlit as st
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import difflib

# Trong run_module1
def run_module1(process_func):
    st.header("Module 1: Gợi ý công việc dựa trên CV")

    # Người dùng nhập văn bản từ web
    resume_text = st.text_area("Nhập văn bản từ CV của bạn:", "")
    
    if st.button("Tìm việc phù hợp"):
        # Gọi hàm từ app.py và hiển thị kết quả
        recommended_jobs = process_func(resume_text)
        
        # Hiển thị kết quả dưới dạng DataFrame với chiều cao và chiều rộng cụ thể
        st.dataframe(recommended_jobs.head(10), height=500, width=1000)  # Hiển thị 10 dòng đầu tiên

# Trong run_module2
def run_module2(career_link, tfidf_matrix):
    st.header("Module 2: Gợi ý công việc dựa trên từ khóa")

    # Input từ khóa từ người dùng
    search_kw = st.text_input("Nhập từ khóa:", "")

    if st.button("Tìm việc phù hợp"):
        recommendations = get_recommendations(search_kw, career_link, tfidf_matrix)

        if recommendations is not None:
            # Hiển thị kết quả dưới dạng DataFrame với chiều cao và chiều rộng cụ thể
            st.dataframe(recommendations.head(10), height=500, width=1000)  # Hiển thị 10 dòng đầu tiên


def get_recommendations(title, career_link, tfidf_matrix):
    n_components = 100
    svd = TruncatedSVD(n_components=n_components)
    tfidf_svd = svd.fit_transform(tfidf_matrix)
    cosine_sim = cosine_similarity(tfidf_svd, tfidf_svd)

    # Tìm job title gần giống nhất
    matching_titles = career_link[career_link['Job Title'].str.contains(title, case=False, na=False)]
    matching_titles_list = list(matching_titles['Job Title'])

    # Tìm các job title gần giống sử dụng difflib
    all_title_list = career_link['Job Title'].tolist()
    find_close_matches = difflib.get_close_matches(title, all_title_list)

    combined_results = list(set(matching_titles_list + find_close_matches))

    if not combined_results:
        st.warning("Xin lỗi, chúng tôi không thể tìm thấy bất kỳ công việc nào gần giống trong các công việc khớp.")
        return None
    else:
        close_match = combined_results[0]
        recommended_jobs = get_recommendations_by_title(close_match, career_link, cosine_sim)
        st.success("Gợi ý công việc dựa trên sự khớp gần nhất:")
        return recommended_jobs

def get_recommendations_by_title(title, career_link, cosine_sim):
    idx = career_link[career_link["Job Title"] == title].index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_similar_indices = [i[0] for i in sim_scores[:31]]

    recommended_jobs = career_link.iloc[top_similar_indices]

    return recommended_jobs
