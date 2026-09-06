import os
import duckdb
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from huggingface_hub import hf_hub_download
from langchain_ibm import WatsonxLLM
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# ======================================
# Konfigurasi Halaman Streamlit
# ======================================
st.set_page_config(
    page_title="Job Insights - IBM Granite",
    page_icon="💼",
    layout="wide"
)

# ======================================
# 1. Validasi Kredensial IBM watsonx
# ======================================
watsonx_apikey = st.secrets.get("WATSONX_APIKEY") or os.getenv("WATSONX_APIKEY")
watsonx_project_id = st.secrets.get("WATSONX_PROJECT_ID") or os.getenv("WATSONX_PROJECT_ID")
watsonx_url = st.secrets.get("WATSONX_URL") or os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

if not watsonx_apikey or not watsonx_project_id:
    st.title("💼 Job Insights - IBM Granite")
    st.error("❌ Kredensial IBM watsonx.ai belum lengkap di Secrets.")
    st.info(
        "Silakan buka **Settings > Secrets** di Streamlit Cloud, lalu tambahkan:\n\n"
        '```toml\n'
        'WATSONX_APIKEY = "API_KEY_IBM_ANDA"\n'
        'WATSONX_PROJECT_ID = "PROJECT_ID_ANDA"\n'
        'WATSONX_URL = "[https://us-south.ml.cloud.ibm.com](https://us-south.ml.cloud.ibm.com)"\n'
        '```'
    )
    st.stop()

DB_PATH = "/tmp/jobs.duckdb"
CACHE_DIR = "/tmp/dataset"

# ======================================
# 2. Inisialisasi Database DuckDB (Hemat RAM)
# ======================================
@st.cache_resource(show_spinner="Sedang menyiapkan database lowongan kerja...")
def get_db():
    # 1. Unduh dataset Parquet
    try:
        local_parquet = hf_hub_download(
            repo_id="barisetiawan51-code/job_dataset",
            filename="job_dataset.parquet",
            repo_type="dataset",
            local_dir=CACHE_DIR
        )
    except Exception:
        local_parquet = "https://huggingface.co/datasets/barisetiawan51-code/job_dataset/resolve/main/job_dataset.parquet"

    # 2. Buat database DuckDB fisik di disk
    if not os.path.exists(DB_PATH):
        raw_conn = duckdb.connect(DB_PATH)
        raw_conn.execute(f"""
            CREATE TABLE jobs AS 
            SELECT * FROM read_parquet('{local_parquet}');
        """)
        raw_conn.close()

    # 3. Hubungkan SQLAlchemy Engine
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
    db, engine = get_db()
except Exception as e:
    st.error(f"❌ Gagal memuat database: {e}")
    st.stop()


# ======================================
# 3. Inisialisasi Model IBM Granite Resmi (Gratis)
# ======================================
@st.cache_resource
def get_llm(apikey: str, project_id: str, url: str):
    parameters = {
        "decoding_method": "greedy",
        "max_new_tokens": 400,
        "temperature": 0.0,
    }
    
    return WatsonxLLM(
        model_id="ibm/granite-3-8b-instruct",
        url=url,
        apikey=apikey,
        project_id=project_id,
        params=parameters
    )

try:
    llm = get_llm(watsonx_apikey, watsonx_project_id, watsonx_url)
except Exception as e:
    st.error(f"❌ Gagal menginisialisasi model IBM Granite: {e}")
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
st.title("💼 Job Insights dengan IBM Granite")
st.write(
    "Tanyakan insight data lowongan pekerjaan, contoh: "
    "*Berapa jumlah lowongan pekerjaan untuk posisi Data Analyst?*"
)

query = st.text_input("Pertanyaan Anda:")

if st.button("Tanyakan", type="primary"):
    if not query.strip():
        st.warning("Silakan masukkan pertanyaan terlebih dahulu.")
    else:
        with st.spinner("IBM Granite sedang menganalisis database..."):
            try:
                result = agent_executor.invoke({"input": query})
                response_text = result.get("output", result) if isinstance(result, dict) else result
                st.success("Jawaban:")
                st.write(response_text)
            except Exception as e:
                st.error(f"Terjadi kendala saat memproses query: {e}")


# ======================================
# 6. Preview Data (5 Baris Pertama)
# ======================================
with st.expander("📊 Pratinjau 5 Baris Pertama Data"):
    try:
        df_preview = pd.read_sql_query("SELECT * FROM jobs LIMIT 5", engine)
        st.dataframe(df_preview, use_container_width=True)
    except Exception as e:
        st.error(f"Gagal memuat pratinjau data: {e}")
