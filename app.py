import os
import duckdb
import streamlit as st
from sqlalchemy import create_engine
from huggingface_hub import hf_hub_download
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# Config halaman Streamlit
st.set_page_config(
    page_title="Job Insights - IBM Granite (Gratis)", 
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
    st.error("❌ HUGGINGFACEHUB_API_TOKEN tidak ditemukan. Harap set di Streamlit Secrets!")
    st.stop()


# ======================================
# 2. Inisialisasi DuckDB Native & LangChain
# ======================================
@st.cache_resource
def get_db_and_conn():
    try:
        # Unduh file Parquet dari Hugging Face ke cache lokal
        local_parquet_path = hf_hub_download(
            repo_id="barisetiawan51-code/job_dataset",
            filename="job_datasett.parquet",
            repo_type="dataset",
            token=hf_token
        )
        
        # Buat koneksi DuckDB native
        raw_conn = duckdb.connect(database=":memory:", read_only=False)
        raw_conn.execute("SET unsafe_disable_etag_checks = true;")
        
        # Buat VIEW 'jobs' langsung di koneksi native ini
        raw_conn.execute(f"CREATE VIEW jobs AS SELECT * FROM read_parquet('{local_parquet_path}');")
        
        # Hubungkan SQLAlchemy engine ke koneksi DuckDB native yang sama
        engine = create_engine("duckdb:///:memory:", creator=lambda: raw_conn)
        db = SQLDatabase(engine)
        
        return db, raw_conn
        
    except Exception as e:
        st.error(f"Gagal mengunduh/memuat Parquet: {e}")
        st.stop()

db, raw_conn = get_db_and_conn()


# ======================================
# 3. Inisialisasi IBM Granite via Hugging Face API
# ======================================
llm = HuggingFaceEndpoint(
    repo_id="ibm-granite/granite-3.0-8b-instruct",
    huggingfacehub_api_token=hf_token,
    max_new_tokens=300,
    temperature=0.1,
)


# ======================================
# 4. Buat SQL Agent
# ======================================
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="zero-shot-react-description",
    verbose=True,
    allow_dangerous_code=True
)


# ======================================
# 5. Streamlit UI
# ======================================
st.title("💼 Job Insights dengan IBM Granite + DuckDB")
st.write("Tanyakan pertanyaan seputar lowongan kerja, contoh: "
         "*berapa banyak postingan job di bidang digital marketing pada tahun 2022?*")

query = st.text_input("Pertanyaan:")

if st.button("Tanyakan", type="primary"):
    if not query.strip():
        st.warning("Silakan masukkan pertanyaan terlebih dahulu.")
    else:
        with st.spinner("IBM Granite sedang menganalisis data via DuckDB..."):
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
        # Gunakan koneksi DuckDB native langsung untuk preview
        df_preview = raw_conn.execute("SELECT * FROM jobs LIMIT 5").df()
        st.dataframe(df_preview)
    except Exception as e:
        st.error(f"Gagal memuat preview data: {e}")
