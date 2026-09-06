import os
import streamlit as st
import pandas as pd
from langchain_community.llms import Replicate
from langchain_experimental.agents import create_pandas_dataframe_agent

# ======================================
# 0. Konfigurasi Halaman Streamlit
# ======================================
st.set_page_config(
    page_title="Job Insights AI",
    page_icon="💼",
    layout="wide"
)

# ======================================
# 1. Autentikasi API Token
# ======================================
# Mengecek token dari Secrets Streamlit Cloud atau Environment OS
replicate_token = None
try:
    if "REPLICATE_API_TOKEN" in st.secrets:
        replicate_token = st.secrets["REPLICATE_API_TOKEN"]
except Exception:
    pass

if not replicate_token:
    replicate_token = os.getenv("REPLICATE_API_TOKEN")

if not replicate_token:
    st.error("❌ `REPLICATE_API_TOKEN` belum disetel.")
    st.info("Buka **Settings > Secrets** di Streamlit Cloud, lalu masukkan:\n`REPLICATE_API_TOKEN = 'token_replicate_anda'`")
    st.stop()

# ======================================
# 2. Load Dataset dari Hugging Face (Aman dari OOM)
# ======================================
# URL file CSV mentah (Raw) dari repositori Hugging Face Anda
DATASET_RAW_URL = "https://huggingface.co/datasets/barisetiawan51-code/job_dataset/raw/main/job_dataset.csv"
SAMPLE_ROWS = 25000  # Batasi 25.000 baris agar muat di RAM 1 GB Streamlit Cloud

@st.cache_data(show_spinner=False)
def load_data(url, nrows):
    # Membaca subset data langsung dari stream URL
    return pd.read_csv(url, nrows=nrows)

with st.spinner(f"Memuat {SAMPLE_ROWS:,} data lowongan kerja..."):
    try:
        df = load_data(DATASET_RAW_URL, SAMPLE_ROWS)
    except Exception as e:
        st.error(f"❌ Gagal memuat data dari Hugging Face: {e}")
        st.info("Pastikan nama file di repo Hugging Face benar-benar 'job_dataset.csv' dan repo berstatus Public.")
        st.stop()

# ======================================
# 3. Inisialisasi Model LLM (IBM Granite)
# ======================================
llm = Replicate(
    model="ibm-granite/granite-3.3-8b-instruct",
    replicate_api_token=replicate_token,
    model_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 300,
        "min_new_tokens": 10,
        "top_p": 0.9,
    },
)

# ======================================
# 4. Inisialisasi Pandas Agent
# ======================================
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    allow_dangerous_code=True,
    handle_parsing_errors=True,
    max_iterations=4
)

# ======================================
# 5. Antarmuka Pengguna (UI)
# ======================================
st.title("💼 Job Insights dengan Granite + Pandas Agent")
st.caption(f"Dataset aktif: {len(df):,} baris data lowongan kerja.")

query = st.text_input(
    "Tanyakan sesuatu tentang data:",
    placeholder="Contoh: Berapa banyak posisi software engineer di data ini?"
)

if st.button("Analisis Data", type="primary"):
    if not query.strip():
        st.warning("Silakan ketik pertanyaan terlebih dahulu.")
    else:
        with st.spinner("Granite sedang menganalisis DataFrame..."):
            try:
                response = agent.invoke(query)
                st.success("Hasil Analisis:")
                
                if isinstance(response, dict) and "output" in response:
                    st.write(response["output"])
                else:
                    st.write(response)
            except Exception as e:
                st.error(f"Terjadi kendala pemrosesan: {e}")

# ======================================
# 6. Sampel Data
# ======================================
with st.expander("📊 Preview 10 Baris Pertama Data"):
    st.dataframe(df.head(10))
