# streamlit_app.py
import streamlit as st
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# === Page Config ===
st.set_page_config(page_title="Fusion RAG Chatbot", layout="wide")
st.title("⚡ Fusion RAG Chatbot (Gemma / Zephyr + FAISS)")

# === Configuration ===
EMBED_MODEL_NAME = "sentence-transformers/sentence-t5-large"
LLM_MODEL_NAME = "google/gemma-2-9b"   # or "HuggingFaceH4/zephyr-7b-beta"
HF_TOKEN = os.getenv("HF_TOKEN", " ")  # set in Streamlit secrets or env

# Local FAISS index (from repo)
FAISS_DIR = Path("faiss_index")
INDEX_NAME = "index"   # expects index.faiss + index.pkl

# === Load Embeddings ===
def load_embeddings():
    """Load embedding model (CPU to save memory)."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )

# === Load FAISS Vector Store (Cached) ===
@st.cache_resource
def load_vector_store(_embeddings):
    """Load FAISS index with safety checks."""
    st.write(f"📁 Loading FAISS from: {FAISS_DIR}")

    if not FAISS_DIR.exists():
        st.error("❌ FAISS directory not found!")
        st.stop()

    try:
        db = FAISS.load_local(
            folder_path=str(FAISS_DIR),
            embeddings=_embeddings,
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True  # Required for pickle
        )
        st.success("✅ FAISS index loaded!")
        return db
    except Exception as e:
        st.error(f"❌ Failed to load FAISS: {e}")
        st.exception(e)
        st.stop()

# === Load LLM (Cached) ===
@st.cache_resource
def load_llm():
    st.info(f"🟢 Loading LLM: {LLM_MODEL_NAME} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME,
            use_auth_token=HF_TOKEN,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            use_auth_token=HF_TOKEN,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        st.success("✅ LLM loaded successfully!")
        return tokenizer, model
    except Exception as e:
        st.error("❌ Failed to load LLM. Common issues:")
        st.markdown(f"""
        - Check your Hugging Face token (needs access to gated model).
        - Do you have internet to pull the model from HF Hub?
        - Error: `{e}`
        """)
        st.stop()

# === Fusion RAG Helpers ===
def generate_queries(original_query: str):
    return [
        original_query,
        f"Explain in detail: {original_query}",
        f"What are the advantages of {original_query}?",
        f"What are the challenges or limitations of {original_query}?",
        f"Give a real-world application of {original_query}"
    ]

def reciprocal_rank_fusion(results_dict, k=60):
    fused_scores = {}
    for query, docs in results_dict.items():
        for rank, doc in enumerate(docs):
            content = doc.page_content.strip()
            if content:
                fused_scores[content] = fused_scores.get(content, 0) + 1 / (rank + k)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

def fusion_rag_answer(query, vectorstore, tokenizer, model):
    if not query.strip():
        return "Please ask a valid question."

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    expanded_queries = generate_queries(query)
    all_results = {}

    with st.spinner("🔍 Retrieving from FAISS ..."):
        for q in expanded_queries:
            docs = retriever.get_relevant_documents(q)
            all_results[q] = docs

    reranked = reciprocal_rank_fusion(all_results)
    if not reranked:
        return "No relevant documents found."

    context = "\n\n---\n\n".join([content for content, _ in reranked[:5]])

    prompt = f"""
You are a helpful AI assistant. Answer only using the context below.

### Context:
{context}

### Question:
{query}

### Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with st.spinner("🧠 Generating answer..."):
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full.split("### Answer:")[-1].strip()
    return answer if answer else "I don't know."

# === Load resources ===
embeddings = load_embeddings()
vectorstore = load_vector_store(embeddings)
tokenizer, model = load_llm()

# === Chat UI ===
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! 👋 Ask me anything based on your knowledge base."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = fusion_rag_answer(user_input, vectorstore, tokenizer, model)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
