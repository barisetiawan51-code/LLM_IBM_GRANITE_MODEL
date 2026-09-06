import os
import duckdb
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from huggingface_hub import hf_hub_download
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# ======================================
# Konfigurasi Halaman Streamlit
# ======================================
st.set_page_config(
    page_title="Job Insights - SQL AI Agent",
    page_icon="💼",
    layout="wide"
)

# ======================================
# 1. Validasi Token Hugging Face
# ======================================
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    st.title("💼 Job Insights - SQL AI Agent")
    st.error("❌ Token `HUGGINGFACEHUB_API_TOKEN` belum ditemukan.")
    st.info(
        "Silakan buka **Settings > Secrets** di Streamlit Cloud, lalu tambahkan:\n\n"
        '```toml\nHUGGINGFACEHUB_API_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"\n```'
    )
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

DB_PATH = "/tmp/jobs.duckdb"
CACHE_DIR = "/tmp/dataset"

# ======================================
# 2. Inisialisasi Database DuckDB (Hemat RAM & Persisten)
# ======================================
@st.cache_resource(show_spinner="Sedang menyiapkan database lowongan kerja...")
def get_db(token: str):
    # 1. Unduh file Parquet dari Hugging Face Hub
    try:
        local_parquet = hf_hub_download(
            repo_id="barisetiawan51-code/job_dataset",
            filename="job_dataset.parquet",
            repo_type="dataset",
            token=token,
            local_dir=CACHE_DIR
        )
    except Exception:
        local_parquet = "https://huggingface.co/datasets/barisetiawan51-code/job_dataset/resolve/main/job_dataset.parquet"

    # 2. Buat database DuckDB fisik jika belum ada
    if not os.path.exists(DB_PATH):
        raw_conn = duckdb.connect(DB_PATH)
        raw_conn.execute(f"""
            CREATE TABLE jobs AS 
            SELECT * FROM read_parquet('{local_parquet}');
        """)
        raw_conn.close()

    # 3. Hubungkan SQLAlchemy Engine ke database DuckDB fisik
    engine = create_engine(f"duckdb:///{DB_PATH}")

    # 4. Hubungkan ke LangChain SQLDatabase
    db = SQLDatabase(
        engine=engine,
        include_tables=["jobs"],
        view_support=True,
        sample_rows_in_table_info=2
    )
    return db, engine

try:
    db, engine = get_db(hf_token)
except Exception as e:
    st.error(f"❌ Gagal memuat database: {e}")
    st.stop()


# ======================================
# 3. Inisialisasi Model LLM (Serverless Supported Model)
# ======================================
@st.cache_resource
def get_llm(token: str):
    # Qwen2.5-Coder-7B-Instruct didukung penuh di Serverless Hugging Face
    # dan memiliki kapabilitas penulisan query SQL yang sangat akurat
    return HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        huggingfacehub_api_token=token,
        max_new_tokens=400,
        temperature=0.01,
        streaming=False,
        timeout=120,
    )

try:
    llm = get_llm(hf_token)
except Exception as e:
    st.error(f"❌ Gagal menginisialisasi model LLM: {e}")
    st.stop()


# ======================================
# 4. Buat Text-to-SQL Agent
# ======================================
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=4,
    allow_dangerous_code=True
)


# ======================================
# 5. Antarmuka (User Interface) Streamlit
# ======================================
st.title("💼 Job Insights dengan SQL AI Agent")
st.write(
    "Tanyakan insight data lowongan pekerjaan, contoh: "
    "*Berapa jumlah lowongan pekerjaan untuk posisi Data Analyst?*"
)

query = st.text_input("Pertanyaan Anda:")

if st.button("Tanyakan", type="primary"):
    if not query.strip():
        st.warning("Silakan masukkan pertanyaan terlebih dahulu.")
    else:
        with st.spinner("AI sedang menganalisis database..."):
            try:
                result = agent_executor.invoke({"input": query})
                response_text = result.get("output", result) if isinstance(result, dict) else result
                st.success("Jawaban:")
                st.write(response_text)
            except Exception as e:
                st.error(f"Terjadi kendala saat memproses query: {e}")


# ======================================
# 6. Preview Sampel Data (5 Baris Pertama)
# ======================================
with st.expander("📊 Pratinjau 5 Baris Pertama Data"):
    try:
        df_preview = pd.read_sql_query("SELECT * FROM jobs LIMIT 5", engine)
        st.dataframe(df_preview, use_container_width=True)
    except Exception as e:
        st.error(f"Gagal memuat pratinjau data: {e}")
