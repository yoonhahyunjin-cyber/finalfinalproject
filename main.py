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
        st.error("❌ 필수 컬럼이 없습니다: Drug 1, Drug 2, I
