import os
import streamlit as st
import duckdb
from langchain_community.llms import Replicate
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# Config halaman Streamlit
st.set_page_config(page_title="Job Insights - IBM Granite", page_icon="💼")

# ======================================
# 1. Ambil API Token dari Streamlit Secrets / Environment Variable
# ======================================
if "REPLICATE_API_TOKEN" in st.secrets:
    replicate_token = st.secrets["REPLICATE_API_TOKEN"]
    os.environ["REPLICATE_API_TOKEN"] = replicate_token
else:
    replicate_token = os.getenv("REPLICATE_API_TOKEN")

if not replicate_token:
    st.error("❌ REPLICATE_API_TOKEN tidak ditemukan. Set di Streamlit Secrets atau .env!")
    st.stop()


# ======================================
# 2. Inisialisasi DuckDB & Konek ke File Parquet
# ======================================
@st.cache_resource
def init_db():
    parquet_path = "https://huggingface.co/datasets/barisetiawan51-code/job_dataset/resolve/main/job_datasett.parquet"
    
    # Buat file database SQLite/DuckDB lokal sementara
    conn = duckdb.connect("jobs_data.db")
    
    # Buat view virtual dari Parquet
    conn.execute(f"CREATE VIEW IF NOT EXISTS jobs AS SELECT * FROM read_parquet('{parquet_path}')")
    conn.close()
    
    # Hubungkan ke SQLDatabase milik LangChain via SQLAlchemy DuckDB dialect
    db = SQLDatabase.from_uri("duckdb:///jobs_data.db")
    return db

try:
    db = init_db()
except Exception as e:
    st.error(f"Gagal memuat dataset Parquet. Detail error: {e}")
    st.stop()


# ======================================
# 3. Inisialisasi LLM
# ======================================
llm = Replicate(
    model="ibm-granite/granite-3.3-8b-instruct",
    replicate_api_token=replicate_token,
    model_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 300,
        "top_p": 0.9,
    },
)


# ======================================
# 4. Buat SQL Agent (Pengganti Pandas Agent)
# ======================================
# SQL Agent membaca skema DuckDB tanpa memuat seluruh file ke RAM
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
st.title("💼 Job Insights dengan Granite + DuckDB")
st.write("Tanyakan pertanyaan seputar lowongan kerja, contoh: "
         "*berapa banyak postingan job di bidang digital marketing pada tahun 2022?*")

query = st.text_input("Pertanyaan:")

if st.button("Tanyakan"):
    if not query.strip():
        st.warning("Silakan masukkan pertanyaan terlebih dahulu.")
    else:
        with st.spinner("Granite sedang menganalisis data via DuckDB..."):
            try:
                # Menjalankan query SQL berbasis agent
                response = agent_executor.run(query)
                st.success("Jawaban Granite Agent:")
                st.write(response)
            except Exception as e:
                st.error(f"Terjadi error saat mengeksekusi pertanyaan: {e}")


# ======================================
# 6. Tampilkan Pratinjau Data (Sampling)
# ======================================
with st.expander("📊 Lihat data awal (5 Baris Pertama)"):
    # Query SQL ringan menggunakan DuckDB langsung
    conn = duckdb.connect("jobs_data.db")
    df_preview = conn.execute("SELECT * FROM jobs LIMIT 5").df()
    conn.close()
    st.dataframe(df_preview)
