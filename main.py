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
st.set_page_config(
    page_title="영양제-약물 군집 유사도 앱",
    page_icon="💊",
    layout="wide",
)
st.title("💊 영양제 → 약물 군집 유사도 탐색기")

st.markdown(
    """
업로드된 **약물 상호작용 데이터(db_drug_interactions.csv)** 를 바탕으로  
비슷한 상호작용 패턴을 보이는 **약물 군집**을 만들고,  
사용자가 입력한 **영양제/성분 이름이 어떤 약물 군집과 의미적으로 가까운지**를 계산합니다.

> ⚠️ 이 결과는 **연구/학습용 참고 정보**이며,  
> 실제 복용/처방 결정은 반드시 **의사·약사와 상의 후** 이루어져야 합니다.
"""
)

# ============================
# 상수 및 경로
# ============================
DATA_PATH = "db_drug_interactions.csv"  # 반드시 저장소에 포함해야 함
NUM_CLUSTERS = 40
TOP_KEYWORDS = 20

# ============================
# 데이터 로드
# ============================
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
def build_drug_corpus(df: pd.DataFrame):
    """
    df에 drug1, drug2, desc 컬럼을 추가하면서
    약물별로 interaction description을 모아 corpus를 만든다.
    """
    df["drug1"] = df["Drug 1"].map(clean_text)
    df["drug2"] = df["Drug 2"].map(clean_text)
    df["desc"] = df["Interaction Description"].map(clean_text)

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
    """
    - 약물 corpus 임베딩
    - KMeans로 NUM_CLUSTERS개 군집 생성
    - 군집별 TF-IDF 키워드 추출
    - 군집 임베딩, 약물-군집 매핑 리턴
    """
    model = load_model()

    # 1. 약물 임베딩
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

    cluster_terms_df = pd.DataFrame(
        {
            "cluster_id": list(range(NUM_CLUSTERS)),
            "top_terms": cluster_keywords,
        }
    )

    # 5. 군집 텍스트 임베딩 (키워드 기반)
    cluster_embs = model.encode(cluster_keywords, normalize_embeddings=True)

    # 6. 약물-군집 매핑
    cluster_assign_df = pd.DataFrame(
        {
            "drug": drug_list,
            "cluster_id": cluster_ids,
        }
    )

    return cluster_terms_df, cluster_embs, cluster_assign_df


# ============================
# 예측 함수
# ============================
def predict_cluster(query_text, cluster_terms_df, cluster_embs, topn=5):
    model = load_model()
    q = str(query_text).strip()

    if not q:
        return pd.DataFrame()

    q_emb = model.encode([q], normalize_embeddings=True)[0]
    sims = cluster_embs @ q_emb  # cosine similarity (정규화된 임베딩)

    order = np.argsort(-sims)[:topn]

    return pd.DataFrame(
        {
            "cluster_id": cluster_terms_df["cluster_id"].iloc[order].tolist(),
            "similarity": sims[order],
            "top_terms": cluster_terms_df["top_terms"].iloc[order].tolist(),
        }
    )


# ============================
# 보조 함수: 군집 예시 가져오기
# ============================
def get_example_drugs(cluster_assign_df, cluster_id, topn=10):
    ex = (
        cluster_assign_df[cluster_assign_df["cluster_id"] == cluster_id]["drug"]
        .head(topn)
        .tolist()
    )
    # 보기 좋게 앞글자만 대문자로
    return [d.title() for d in ex]


def get_example_interactions(df, drug_clean_name, max_n=3):
    mask = (df["drug1"] == drug_clean_name) | (df["drug2"] == drug_clean_name)
    rows = df[mask].head(max_n)
    examples = rows["Interaction Description"].dropna().tolist()
    return examples


# ============================
# 메인 UI
# ============================
df = load_raw_data()
drug_list, drug_corpus = build_drug_corpus(df)
cluster_terms_df, cluster_embs, cluster_assign_df = build_clusters(drug_list, drug_corpus)

st.success("✅ 데이터 로딩 & 모델 준비 완료!")

# ----- 사이드바 -----
with st.sidebar:
    st.header("⚙️ 검색 옵션")

    user_input = st.text_input("영양제 / 성분 이름 입력", "홍삼")
    topn = st.slider("Top-N 군집", 3, 15, 5)
    run = st.button("유사도 계산하기")

st.markdown("---")

# ----- 메인 영역 -----
if not run:
    st.info("왼쪽 사이드바에서 **영양제/성분 이름**을 입력하고 '유사도 계산하기' 버튼을 눌러주세요.")
else:
    result = predict_cluster(user_input, cluster_terms_df, cluster_embs, topn=topn)

    if result.empty:
        st.warning("입력 값이 비어 있습니다. 영양제나 성분 이름을 입력해 주세요.")
        st.stop()

    st.subheader(f"🔎 '{user_input}'와 의미적으로 가까운 약물 군집 Top-{topn}")

    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(
        ["요약", "군집별 특징 & 예시 약물", "상호작용 설명 예시", "방법 설명"]
    )

    # ---------------- 탭 1: 요약 ----------------
    with tab1:
        st.markdown("### ✨ 유사도 요약")

        # 상위 1개 군집 강조
        top_cluster = int(result.iloc[0]["cluster_id"])
        top_sim = float(result.iloc[0]["similarity"])

        c1, c2 = st.columns(2)
        with c1:
            st.metric("가장 가까운 군집 ID", top_cluster)
        with c2:
            st.metric("해당 군집과의 유사도", f"{top_sim:.3f}")

        st.markdown("#### 📊 Top-N 군집 유사도")
        chart_df = result[["cluster_id", "similarity"]].copy()
        chart_df = chart_df.set_index("cluster_id")
        st.bar_chart(chart_df)

        st.markdown(
            """
- 막대 그래프의 **높을수록** 입력한 영양제/성분과 의미적으로 더 가까운 군집입니다.  
- 아래 다른 탭에서 각 군집이 어떤 **약물/상호작용 패턴**을 대표하는지 자세히 볼 수 있습니다.
"""
        )

    # ---------------- 탭 2: 군집별 특징 & 예시 약물 ----------------
    with tab2:
        st.markdown("### 🧬 군집별 키워드 & 예시 약물")

        for _, row in result.iterrows():
            cid = int(row["cluster_id"])
            sim = float(row["similarity"])
            keywords = row["top_terms"].split(", ")[:8]  # 상위 8개만 보기 좋게

            with st.expander(f"군집 {cid}  |  유사도 {sim:.3f}"):
                c1, c2 = st.columns([2, 1])

                with c1:
                    st.markdown("**대표 키워드 (TF-IDF):**")
                    st.write(", ".join(keywords))

                with c2:
                    example_drugs = get_example_drugs(cluster_assign_df, cid, topn=8)
                    if example_drugs:
                        st.markdown("**이 군집의 예시 약물:**")
                        st.write(", ".join(example_drugs))
                    else:
                        st.write("예시 약물을 가져올 수 없습니다.")

    # ---------------- 탭 3: 상호작용 설명 예시 ----------------
    with tab3:
        st.markdown("### ⚗️ 상호작용 설명 예시")

        # 어떤 군집을 자세히 볼지 선택
        cid_options = result["cluster_id"].astype(int).tolist()
        selected_cid = st.selectbox("상세히 보고 싶은 군집 선택", cid_options)

        example_drugs = get_example_drugs(cluster_assign_df, selected_cid, topn=5)
        if not example_drugs:
            st.write("이 군집에 예시로 보여줄 약물이 없습니다.")
        else:
            st.markdown(f"**군집 {selected_cid} 예시 약물들:**")
            st.write(", ".join(example_drugs))

            st.markdown("---")
            st.markdown("#### 상호작용 설명 일부 예시")

            # 첫 번째 예시 약물 기준으로 상호작용 설명 몇 개 가져오기
            first_clean = example_drugs[0].lower()
            clean_name = clean_text(first_clean)
            ex_interactions = get_example_interactions(df, clean_name, max_n=5)

            if ex_interactions:
                for i, desc in enumerate(ex_interactions, 1):
                    st.markdown(f"**예시 {i}.** {desc}")
            else:
                st.write("상호작용 설명 예시를 찾지 못했습니다.")

    # ---------------- 탭 4: 방법 설명 ----------------
    with tab4:
        st.markdown("### 🧪 이 앱은 어떻게 동작하나요?")

        st.markdown(
            """
1. **데이터 준비**  
   - `db_drug_interactions.csv`에서 `Drug 1`, `Drug 2`, `Interaction Description`을 읽어옵니다.  
   - 간단한 텍스트 전처리 후, 약물별로 interaction description을 합쳐 **약물 단위 corpus**를 만듭니다.

2. **군집 형성**  
   - 다국어 문장 임베딩 모델(`paraphrase-multilingual-MiniLM`)로 각 약물의 corpus를 벡터화합니다.  
   - KMeans로 약물들을 **{NUM_CLUSTERS}개의 군집**으로 묶습니다.  
   - 각 군집 안의 텍스트를 모아 TF-IDF를 적용해 **대표 키워드**를 뽑습니다.

3. **영양제/성분 → 군집 유사도**  
   - 사용자가 입력한 문자열을 동일한 임베딩 모델로 벡터화합니다.  
   - 군집 임베딩(대표 키워드 기반)과 코사인 유사도를 계산해,  
     **의미적으로 가까운 군집 Top-N**을 보여줍니다.

4. **해석 시 주의점**  
   - 실제 임상적 상호작용 위험도와 **정확히 일치하지 않을 수 있습니다.**  
   - 데이터에 없는 영양제/성분은 잘 매핑되지 않을 수 있습니다.  
   - 이 결과는 **학습/연구용 참고 정보**로 활용하고,  
     실제 복용/처방은 반드시 전문가 상담 후 결정해야 합니다.
"""
        )
