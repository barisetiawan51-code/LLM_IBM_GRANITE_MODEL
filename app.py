import os
import gc
import pandas as pd
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
    st.error("❌ Token `HUGGINGFACEHUB_API_TOKEN` belum disetel.")
    st.info(
        "Buka **Settings > Secrets** di Streamlit Cloud dan tambahkan:\n\n"
        '```toml\nHUGGINGFACEHUB_API_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"\n```'
    )
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

DB_PATH = "/tmp/jobs.db"

# ======================================
# 2. Inisialisasi Database SQLite Hemat RAM
# ======================================
@st.cache_resource(show_spinner="Sedang menyiapkan database hemat memori...")
def get_db(token: str):
    # Jika database lokal file sudah pernah dibuat di session ini, pakai langsung
    engine = create_engine(f"sqlite:///{DB_PATH}")

    if not os.path.exists(DB_PATH):
        try:
            local_parquet_path = hf_hub_download(
                repo_id="barisetiawan51-code/job_dataset",
                filename="job_dataset.parquet",
                repo_type="dataset",
                token=token,
                local_dir="/tmp/hf_cache"
            )
            df = pd.read_parquet(local_parquet_path)
        except Exception:
            url = "https://huggingface.co/datasets/barisetiawan51-code/job_dataset/resolve/main/job_dataset.parquet"
            df = pd.read_parquet(url, storage_options={"Authorization": f"Bearer {token}"})

        # Masukkan ke database file SQLite bertahap (chunksize) agar RAM tidak lonjak
        df.to_sql("jobs", con=engine, index=False, if_exists="replace", chunksize=5000)

        # Hapus DataFrame dari RAM dan bersihkan memori
        del df
        gc.collect()

    # Hubungkan ke LangChain SQLDatabase
    db = SQLDatabase(
        engine=engine,
        include_tables=["jobs"],
        sample_rows_in_table_info=2  # Hemat token prompt & hemat memori
    )
    return db, engine

try:
    db, engine = get_db(hf_token)
except Exception as e:
    st.error(f"❌ Gagal memuat database: {e}")
    st.stop()


# ======================================
# 3. Inisialisasi IBM Granite Model
# ======================================
@st.cache_resource
def get_llm(token: str):
    return HuggingFaceEndpoint(
        repo_id="ibm-granite/granite-3.0-8b-instruct",
        huggingfacehub_api_token=token,
        max_new_tokens=300,       # Dibatasi agar respons lebih cepat dan ringan
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
# 4. Buat SQL Agent Ringan
# ======================================
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=4,            # Batasi iterasi agar tidak looping lama
    allow_dangerous_code=True
)


# ======================================
# 5. Tampilan Streamlit UI
# ======================================
st.title("💼 Job Insights dengan IBM Granite")
st.write("Tanyakan pertanyaan seputar data lowongan kerja.")

query = st.text_input("Masukkan pertanyaan:")

if st.button("Tanyakan", type="primary"):
    if not query.strip():
        st.warning("Silakan masukkan pertanyaan terlebih dahulu.")
    else:
        with st.spinner("IBM Granite sedang menganalisis..."):
            try:
                result = agent_executor.invoke({"input": query})
                response_text = result.get("output", result) if isinstance(result, dict) else result
                st.success("Jawaban:")
                st.write(response_text)
            except Exception as e:
                st.error(f"Gagal memproses pertanyaan: {e}")


# ======================================
# 6. Preview Data (Sampling Ringan)
# ======================================
with st.expander("📊 Lihat 5 Baris Pertama"):
    try:
        df_preview = pd.read_sql_query("SELECT * FROM jobs LIMIT 5", engine)
        st.dataframe(df_preview, use_container_width=True)
    except Exception as e:
        st.error(f"Gagal memuat preview data: {e}")
