import os
import streamlit as st
from sqlalchemy import create_engine
from huggingface_hub import hf_hub_download
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

st.set_page_config(
    page_title="Job Insights - IBM Granite",
    page_icon="💼",
    layout="wide"
)

# ======================================
# 1. Ambil API Token Hugging Face
# ======================================
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    st.title("💼 Job Insights - IBM Granite")
    st.error("❌ Token `HUGGINGFACEHUB_API_TOKEN` belum disetel di Secrets.")
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token


# ======================================
# 2. Setup DuckDB Langsung ke File Parquet (Zero-Copy Memory)
# ======================================
@st.cache_resource(show_spinner="Menyiapkan koneksi dataset...")
def get_db(token: str):
    # Unduh file Parquet saja (disimpan di disk /tmp, tidak masuk RAM)
    local_parquet = hf_hub_download(
        repo_id="barisetiawan51-code/job_dataset",
        filename="job_dataset.parquet",
        repo_type="dataset",
        token=token,
        local_dir="/tmp/dataset"
    )

    # Buat engine DuckDB
    engine = create_engine("duckdb:///:memory:")

    # Buat VIEW yang membaca langsung file Parquet tanpa menduplikasi data ke RAM
    with engine.connect() as conn:
        conn.exec_driver_sql(f"""
            CREATE VIEW jobs AS 
            SELECT * FROM read_parquet('{local_parquet}');
        """)

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
    st.error(f"Gagal memuat database: {e}")
    st.stop()


# ======================================
# 3. Model IBM Granite
# ======================================
@st.cache_resource
def get_llm(token: str):
    return HuggingFaceEndpoint(
        repo_id="ibm-granite/granite-3.0-8b-instruct",
        huggingfacehub_api_token=token,
        max_new_tokens=300,
        temperature=0.01,
        streaming=False,
        task="text-generation",
    )

try:
    llm = get_llm(hf_token)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()


# ======================================
# 4. SQL Agent
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
# 5. UI Streamlit
# ======================================
st.title("💼 Job Insights - IBM Granite")
st.write("Tanyakan statistik atau informasi dari data lowongan kerja.")

query = st.text_input("Pertanyaan:")

if st.button("Tanyakan", type="primary"):
    if not query.strip():
        st.warning("Masukkan pertanyaan terlebih dahulu.")
    else:
        with st.spinner("Menganalisis..."):
            try:
                res = agent_executor.invoke({"input": query})
                output = res.get("output", res) if isinstance(res, dict) else res
                st.success("Jawaban:")
                st.write(output)
            except Exception as e:
                st.error(f"Terjadi error: {e}")


# ======================================
# 6. Preview Data Ringan
# ======================================
with st.expander("📊 Preview 5 Baris Data"):
    try:
        import pandas as pd
        df_preview = pd.read_sql_query("SELECT * FROM jobs LIMIT 5", engine)
        st.dataframe(df_preview, use_container_width=True)
    except Exception as e:
        st.error(f"Gagal preview: {e}")
