import streamlit as st
import pandas as pd
from sentence-transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# ====== SETUP ======
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

model = load_model()

def build_faiss_index(texts):
    embeddings = model.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

def retrieve(query, index, df, top_k=None):
    return df  # Optional: Tambahkan top_k filtering jika perlu

def generate_answer(query, context, api_key):
    openai.api_key = api_key
    system_message = "Kamu adalah asisten cerdas yang menjawab pertanyaan berdasarkan data yang diberikan."
    user_message = f"""
    Pertanyaan: {query}

    Data yang relevan:
    {context}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message["content"].strip()

# ====== STREAMLIT UI ======
st.title("📊 RAG CSV Umum (Tanpa Struktur Khusus)")

# ====== SIDEBAR ======
st.sidebar.header("🔧 Pengaturan")

uploaded_file = st.sidebar.file_uploader("📂 Upload file CSV", type="csv")
input_api_key = st.sidebar.text_input("🔑 Masukkan OpenAI API Key", type="password")
activate_api = st.sidebar.button("🔒 Aktifkan API Key")

# Inisialisasi session state
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "history" not in st.session_state:
    st.session_state.history = []

if activate_api and input_api_key:
    st.session_state.api_key = input_api_key
    st.sidebar.success("✅ API Key aktif!")

# Tombol reset riwayat
if st.sidebar.button("🗑️ Hapus Riwayat"):
    st.session_state.history = []
    st.sidebar.success("Riwayat berhasil dihapus!")

# ====== MAIN INPUT ======
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🗂️ Pilih Kolom untuk Dipelajari oleh RAG")
    selected_columns = st.multiselect(
        "Pilih kolom yang ingin diproses:",
        options=df.columns.tolist(),
        default=df.columns.tolist()
    )

    if not selected_columns:
        st.warning("⚠️ Harap pilih setidaknya satu kolom.")
        st.stop()

    # Tampilkan preview data dari kolom terpilih
    st.write("📄 Pratinjau Data Terpilih")
    st.dataframe(df[selected_columns])

    def transform_data(df, selected_columns):
        df["text"] = df[selected_columns].astype(str).agg(" | ".join, axis=1)
        return df

    # Input pertanyaan hanya muncul jika kolom telah dipilih
    query = st.text_input("❓ Masukkan pertanyaan Anda")
    run_query = st.button("🚀 Jawab Pertanyaan")

    if run_query and st.session_state.api_key:
        try:
            df = transform_data(df, selected_columns)
            index, _ = build_faiss_index(df["text"].tolist())

            with st.spinner("🔍 Mencari data relevan..."):
                results = retrieve(query, index, df)
                context = "\n".join(results["text"].tolist())

            with st.spinner("🧠 Menghasilkan jawaban..."):
                answer = generate_answer(query, context, st.session_state.api_key)

            st.subheader("💬 Jawaban:")
            st.success(answer)
            st.session_state.history.append((query, answer))

        except Exception as e:
            st.error(f"Terjadi error: {str(e)}")
    elif run_query and not st.session_state.api_key:
        st.warning("🔐 Anda harus mengaktifkan API Key terlebih dahulu.")
    else:
        st.warning("📂 Silakan upload file CSV terlebih dahulu.")

# ====== HISTORY ======
if st.session_state.history:
    st.subheader("🕘 Riwayat Pertanyaan dan Jawaban")
    for i, (q, a) in enumerate(reversed(st.session_state.history[-5:]), 1):
        with st.expander(f"❓ Pertanyaan #{len(st.session_state.history)-i+1}: {q}"):
             st.markdown(f"💬 **Jawaban:** {a}")
