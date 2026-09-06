import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from huggingface_hub import hf_hub_download
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# Config halaman Streamlit
st.set_page_config(
    page_title="Job Insights - IBM Granite",
    page_icon="💼",
    layout="wide"
)

# ======================================
# 1. Ambil API Token Hugging Face
# ======================================
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
else:
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    st.error("❌ HUGGINGFACEHUB_API_TOKEN tidak ditemukan. Harap tambahkan di Streamlit Secrets atau environment variable!")
    st.stop()


# ======================================
# 2. Inisialisasi DuckDB via SQLAlchemy (:memory:)
# ======================================
@st.cache_resource(show_spinner="Sedang mengunduh dataset dan menyiapkan database...")
def get_db(token: str):
    # 1. Unduh file Parquet dari Hugging Face Hub
    local_parquet_path = hf_hub_download(
        repo_id="barisetiawan51-code/job_dataset",
        filename="job_dataset.parquet",
        repo_type="dataset",
        token=token
    )

    # 2. Buat in-memory DuckDB engine langsung via SQLAlchemy
    # Menggunakan :memory: menghindari konflik konfigurasi file dan file lock
    engine = create_engine("duckdb:///:memory:")

    # 3. Buat tabel jobs langsung menggunakan koneksi engine yang sama
    with engine.connect() as conn:
        conn.exec_driver_sql(f"""
            CREATE TABLE jobs AS 
            SELECT * FROM read_parquet('{local_parquet_path}');
        """)

    # 4. Hubungkan ke LangChain SQLDatabase
    db = SQLDatabase(
        engine=engine,
        include_tables=["jobs"],
        view_support=True
    )
    return db, engine

# Jalankan inisialisasi database
try:
    db, engine = get_db(hf_token)
except Exception as e:
    st.error(f"❌ Gagal memuat database dari Hugging Face: {e}")
    st.stop()


# ======================================
# 3. Inisialisasi IBM Granite via Hugging Face API
# ======================================
@st.cache_resource
def get_llm(token: str):
    # streaming=False dan task='text-generation' mencegah error 'generator raised StopIteration'
    return HuggingFaceEndpoint(
        repo_id="ibm-granite/granite-3.0-8b-instruct",
        huggingfacehub_api_token=token,
        max_new_tokens=512,
        temperature=0.01,
        streaming=False,
        task="text-generation",
    )

try:
    llm = get_llm(hf_token)
except Exception as e:
    st.error(f"❌ Gagal menginisialisasi model LLM: {e}")
    st.stop()


# ======================================
# 4. Buat SQL Agent
# ======================================
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    allow_dangerous_code=True
)


# ======================================
# 5. Streamlit UI
# ======================================
st.title("💼 Job Insights dengan IBM Granite")
st.write(
    "Tanyakan pertanyaan seputar lowongan kerja, contoh: "
    "*berapa banyak postingan job di bidang digital marketing pada tahun 2022?*"
)

query = st.text_input("Pertanyaan:")

if st.button("Tanyakan", type="primary"):
    if not query.strip():
        st.warning("Silakan masukkan pertanyaan terlebih dahulu.")
    else:
        with st.spinner("IBM Granite sedang menganalisis data..."):
            try:
                result = agent_executor.invoke({"input": query})
                response_text = result.get("output", result) if isinstance(result, dict) else result
                st.success("Jawaban IBM Granite Agent:")
                st.write(response_text)
            except Exception as e:
                st.error(f"Terjadi error saat mengeksekusi pertanyaan: {e}")


# ======================================
# 6. Tampilkan Pratinjau Data (Sampling)
# ======================================
with st.expander("📊 Lihat data awal (5 Baris Pertama)"):
    try:
        df_preview = pd.read_sql_query("SELECT * FROM jobs LIMIT 5", engine)
        st.dataframe(df_preview, use_container_width=True)
    except Exception as e:
        st.error(f"Gagal memuat preview data: {e}")
