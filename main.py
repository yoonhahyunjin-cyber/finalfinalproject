import os
import re
import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================
# Streamlit 기본 설정
# ============================
st.set_page_config(page_title="영양제-약물 군집 유사도 앱", layout="wide")
st.title("💊 영양제 → 약물 군집 유사도 탐색기")

st.markdown("""
업로드된 약물 상호작용(db_drug_interactions.csv) 데이터를 기반으로 약물 군집을 만들고,  
영양제 입력 시 어떤 약물 군집과 의미적으로 가장 가까운지 계산해줍니다.
""")


# ============================
# 데이터 로드
# ============================
DATA_PATH = "db_drug_interactions.csv"  # 반드시 저장소에 포함해야 함
NUM_CLUSTERS = 40
TOP_KEYWORDS = 20

@st.cache_data
def load_raw_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ '{DATA_PATH}' 파일을 찾지 못했습니다. 저장소에 포함시키세요.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    
    required = {"Drug 1", "Drug 2", "Interaction Description"}
    if not required.issubset(df.columns):
        st.error("❌ 필수 컬럼이 없습니다: Drug 1, Drug 2, Interaction Description")
        st.stop()
    return df


# ============================
# 전처리 함수
# ============================
def clean_text(t: str) -> str:
    if pd.isna(t):
        return ""
    t = str(t).lower()
    t = re.sub(r"[^a-zA-Z0-9가-힣\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


@st.cache_data
def build_drug_corpus(df):
    df["drug1"] = df["Drug 1"].map(clean_text)
    df["drug2"] = df["Drug 2"].map(clean_text)
    df["desc"]  = df["Interaction Description"].map(clean_text)

    drug_texts = {}

    for _, row in df.iterrows():
        d1, d2, desc = row["drug1"], row["drug2"], row["desc"]

        if d1:
            drug_texts.setdefault(d1, []).append(desc)
        if d2:
            drug_texts.setdefault(d2, []).append(desc)

    drug_list, drug_corpus = [], []
    for drug, texts in drug_texts.items():
        drug_list.append(drug)
        drug_corpus.append(" ".join(texts))

    return drug_list, drug_corpus


# ============================
# 임베딩 모델 로딩
# ============================
@st.cache_resource
def load_model():
    MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return SentenceTransformer(MODEL)


# ============================
# 약물 군집 만들기
# ============================
@st.cache_resource
def build_clusters(drug_list, drug_corpus):
    model = load_model()

    # 1. 임베딩
    drug_embs = model.encode(drug_corpus, normalize_embeddings=True)

    # 2. KMeans 클러스터링
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(drug_embs)

    # 3. 군집별 텍스트 모으기
    cluster_texts = []
    for c in range(NUM_CLUSTERS):
        texts = [drug_corpus[i] for i in range(len(drug_list)) if cluster_ids[i] == c]
        merged = " ".join(texts) if texts else "no data"
        cluster_texts.append(merged)

    # 4. TF-IDF 키워드 추출
    tfidf = TfidfVectorizer(max_features=1000, stop_words="english")
    X = tfidf.fit_transform(cluster_texts)
    terms = tfidf.get_feature_names_out()

    cluster_keywords = []
    for i in range(NUM_CLUSTERS):
        row = X[i].toarray()[0]
        idx = row.argsort()[::-1][:TOP_KEYWORDS]
        keywords = ", ".join([terms[j] for j in idx])
        cluster_keywords.append(keywords)

    cluster_terms_df = pd.DataFrame({
        "cluster_id": list(range(NUM_CLUSTERS)),
        "top_terms": cluster_keywords
    })

    # 5. 군집 텍스트 임베딩
    cluster_embs = model.encode(cluster_keywords, normalize_embeddings=True)

    return cluster_terms_df, cluster_embs


# ============================
# 예측 함수
# ============================
def predict_cluster(name, cluster_terms_df, cluster_embs, topn=5):
    model = load_model()
    q = str(name).strip()

    if not q:
        return pd.DataFrame()

    q_emb = model.encode([q], normalize_embeddings=True)[0]
    sims = cluster_embs @ q_emb

    order = np.argsort(-sims)[:topn]

    return pd.DataFrame({
        "cluster_id": cluster_terms_df["cluster_id"].iloc[order].tolist(),
        "similarity": sims[order],
        "top_terms": cluster_terms_df["top_terms"].iloc[order].tolist()
    })


# ============================
# 메인 UI
# ============================
df = load_raw_data()
drug_list, drug_corpus = build_drug_corpus(df)
cluster_terms_df, cluster_embs = build_clusters(drug_list, drug_corpus)

st.success("데이터 로딩 & 모델 준비 완료!")

with st.sidebar:
    st.header("⚙️ 옵션")
    user_input = st.text_input("영양제 / 성분 이름 입력", "홍삼")
    topn = st.slider("Top-N 군집", 3, 15, 5)
    run = st.button("유사도 계산하기")

st.markdown("---")

if run:
    result = predict_cluster(user_input, cluster_terms_df, cluster_embs, topn=topn)

    st.subheader(f"🔎 '{user_input}'와 가장 유사한 약물 군집 Top-{topn}")
    st.dataframe(result)

    st.bar_chart(result.set_index("cluster_id")["similarity"])
else:
    st.info("왼쪽 사이드바에서 영양제를 입력하고 '유사도 계산하기' 버튼을 누르세요.")
