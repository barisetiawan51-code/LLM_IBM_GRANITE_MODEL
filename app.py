import os
import streamlit as st
import pandas as pd
from datasets import load_dataset
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
# 1. Ambil API Token dari Secrets / Environment
# ======================================
replicate_token = st.secrets.get("REPLICATE_API_TOKEN") or os.getenv("REPLICATE_API_TOKEN")
hf_token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

if not replicate_token:
    st.error("❌ `REPLICATE_API_TOKEN` belum disetel di Streamlit Secrets.")
    st.stop()

# ======================================
# 2. Load Dataset dari Hugging Face
# ======================================
# GANTI dengan path repository dataset Hugging Face Anda
# Contoh: "username_anda/nama_dataset"
HF_DATASET_REPO = "barisetiawan51-code/job_dataset"

@st.cache_data(show_spinner=False)
def load_data():
    # Jika dataset public, hf_token bernilai None dan tetap bisa di-load
    dataset = load_dataset(
        HF_DATASET_REPO, 
        split="train", 
        token=hf_token
    )
    return dataset.to_pandas()

with st.spinner("Mengunduh dan memuat dataset dari Hugging Face..."):
    try:
        df = load_data()
    except Exception as e:
        st.error(f"❌ Gagal memuat dataset dari Hugging Face: {e}")
        st.stop()

# ======================================
# 3. Inisialisasi LLM (Replicate)
# ======================================
llm = Replicate(
    model="ibm-granite/granite-3.3-8b-instruct",
    replicate_api_token=replicate_token,
    model_kwargs={
        "temperature": 0.2,
        "max_new_tokens": 300,
        "min_new_tokens": 50,
        "top_p": 0.9,
        "top_k": 50,
    },
)

# ======================================
# 4. Inisialisasi Pandas DataFrame Agent
# ======================================
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    allow_dangerous_code=True
)

# ======================================
# 5. Tampilan Antarmuka (Streamlit UI)
# ======================================
st.title("💼 Job Insights dengan Granite + Pandas Agent")
st.write(
    "Tanyakan pertanyaan seputar data lowongan kerja, misalnya: "
    "*'Berapa banyak postingan job di bidang digital marketing pada tahun 2022?'*"
)

query = st.text_input("Pertanyaan:", placeholder="Tuliskan pertanyaan di sini...")

if st.button("Tanyakan", type="primary"):
    if not query.strip():
        st.warning("⚠️ Silakan masukkan pertanyaan terlebih dahulu.")
    else:
        with st.spinner("Granite sedang menganalisis data..."):
            try:
                # Menggunakan invoke() standar LangChain terbaru
                response = agent.invoke(query)
                st.success("Jawaban Granite + Pandas Agent:")
                
                # Menangani output baik berupa string maupun dictionary
                if isinstance(response, dict) and "output" in response:
                    st.write(response["output"])
                else:
                    st.write(response)
            except Exception as e:
                st.error(f"Terjadi error saat memproses pertanyaan: {e}")

# ======================================
# 6. Eksplorasi Data Awal
# ======================================
with st.expander("📊 Lihat Data Awal (5 Baris Pertama)"):
    st.write(f"Total baris data: **{len(df)}** | Total kolom: **{len(df.columns)}**")
    st.dataframe(df.head())
