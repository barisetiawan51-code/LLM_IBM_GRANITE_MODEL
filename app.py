import os
import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import fsspec
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
replicate_token = None
try:
    if "REPLICATE_API_TOKEN" in st.secrets:
        replicate_token = st.secrets["REPLICATE_API_TOKEN"]
except Exception:
    pass

if not replicate_token:
    replicate_token = os.getenv("REPLICATE_API_TOKEN")

if not replicate_token:
    st.error("❌ `REPLICATE_API_TOKEN` belum disetel di Streamlit Secrets.")
    st.stop()

# ======================================
# 2. Load Parquet Parsial via fsspec (Hemat RAM)
# ======================================
PARQUET_URL = "https://huggingface.co/datasets/barisetiawan51-code/job_dataset/resolve/main/job_dataset.parquet"
SAMPLE_ROWS = 20000

@st.cache_data(show_spinner=False)
def load_data(url, limit):
    # fsspec menangani protokol HTTP/HTTPS tanpa memuat file 726 MB secara utuh
    with fsspec.open(url, mode="rb") as f:
        parquet_file = pq.ParquetFile(f)
        
        # Baca row group awal saja sesuai kebutuhan sampel
        batches = []
        total_rows = 0
        for i in range(parquet_file.num_row_groups):
            table_batch = parquet_file.read_row_group(i)
            df_batch = table_batch.to_pandas()
            batches.append(df_batch)
            total_rows += len(df_batch)
            if total_rows >= limit:
                break
                
        df_final = pd.concat(batches, ignore_index=True)
        return df_final.head(limit)

with st.spinner("Mengambil sebagian data lowongan kerja dari Hugging Face..."):
    try:
        df = load_data(PARQUET_URL, SAMPLE_ROWS)
    except Exception as e:
        st.error(f"❌ Gagal memuat data: {e}")
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
# 5. Antarmuka Pengguna (Streamlit UI)
# ======================================
st.title("💼 Job Insights dengan Granite + Pandas Agent")
st.caption(f"Dataset aktif: {len(df):,} baris data lowongan kerja.")

query = st.text_input(
    "Tanyakan sesuatu tentang data:",
    placeholder="Contoh: Berapa banyak postingan pekerjaan untuk posisi data analyst?"
)

if st.button("Analisis Data", type="primary"):
    if not query.strip():
        st.warning("⚠️ Silakan ketik pertanyaan terlebih dahulu.")
    else:
        with st.spinner("Granite sedang menganalisis data..."):
            try:
                response = agent.invoke(query)
                st.success("Hasil Analisis:")
                
                if isinstance(response, dict) and "output" in response:
                    st.write(response["output"])
                else:
                    st.write(response)
            except Exception as e:
                st.error(f"Terjadi kendala saat analisis: {e}")

# ======================================
# 6. Preview Data
# ======================================
with st.expander("📊 Preview 10 Baris Pertama"):
    st.dataframe(df.head(10))
