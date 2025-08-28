# app.py
import os
from pathlib import Path
import time
import io

import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pypdf import PdfReader

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(page_title="⚡ Fusion RAG Chatbot", page_icon="🤖", layout="wide")
st.title("⚡ Fusion RAG Chatbot (Local RAG + FAISS)")

# =========================
# Secrets / Tokens
# =========================
HF_TOKEN = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "").strip()

if not HF_TOKEN:
    st.warning("No Hugging Face token found. Set `HF_TOKEN` in Streamlit secrets or env.")
    st.info("For public models like TinyLlama, you can leave it blank.")
    # We'll proceed anyway since TinyLlama is public

# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    st.header("⚙️ Settings")

    # Embedding model
    embedding_model_name = st.text_input(
        "Embedding model",
        value="sentence-transformers/all-MiniLM-L6-v2",
        help="Used for semantic search in FAISS."
    )

    # LLM model - CHANGED to TinyLlama (works on Streamlit Cloud)
    llm_model_name = st.text_input(
        "LLM model (Transformers)",
        value="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Use small models like 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' for Streamlit Cloud"
    )

    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    max_new_tokens = st.slider("Max new tokens", 64, 512, 256, 32)
    k_retrieval = st.slider("Top-k chunks", 1, 6, 3, 1)

    show_chunks = st.checkbox("Show retrieved chunks", value=False)

    st.divider()
    st.subheader("📄 Load Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDFs or text files", type=["pdf", "txt"], accept_multiple_files=True
    )
    persist_dir = st.text_input(
        "FAISS persist directory (optional)",
        value="faiss_index",
        help="Index will be saved here between sessions."
    )
    rebuild_index = st.checkbox("Rebuild index on upload", value=True)


# =========================
# Utilities
# =========================
@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}  # Fast enough on CPU
    )


def _read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        try:
            text = page.extract_text()
            if text:
                texts.append(text)
        except Exception:
            continue
    return "\n".join(texts)


def _read_txt(file_bytes: bytes) -> str:
    for enc in ["utf-8", "latin-1"]:
        try:
            return file_bytes.decode(enc)
        except Exception:
            continue
    return ""


def load_documents(files) -> list[dict]:
    """Return list of dicts: {'source': filename, 'text': content}"""
    docs = []
    for f in files or []:
        name = f.name
        content = f.read()
        if name.lower().endswith(".pdf"):
            text = _read_pdf(content)
        else:
            text = _read_txt(content)
        text = text.strip()
        if text:
            docs.append({"source": name, "text": text})
    return docs


def chunk_documents(raw_docs: list[dict], chunk_size=512, chunk_overlap=64):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for d in raw_docs:
        for i, c in enumerate(splitter.split_text(d["text"])):
            chunks.append({
                "source": d["source"],
                "chunk_id": f"{d['source']}#{i}",
                "text": c.strip()
            })
    return chunks


@st.cache_resource(show_spinner=True)
def build_or_load_faiss(chunks: list[dict], embeddings, persist_path: str | None):
    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks]

    if persist_path:
        path = Path(persist_path)
        if not path.exists() or rebuild_index:
            st.info("🧠 Building new FAISS index...")
            vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
            path.mkdir(parents=True, exist_ok=True)
            vs.save_local(str(path))
            st.success(f"✅ Saved FAISS index to `{persist_path}`")
        else:
            st.info(f"📁 Loading FAISS index from `{persist_path}`...")
            vs = FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    else:
        vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    return vs


# =========================
# Load LLM Model
# =========================
@st.cache_resource(show_spinner="🚀 Loading LLM...")
def load_local_model(model_name: str, token: str | None):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Simple chat template for TinyLlama
        if not tokenizer.chat_template:
            tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            torch_dtype=torch.float32,  # CPU-safe; use float16 if GPU
            device_map=None  # Let PyTorch handle it
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            model = model.to(device).eval()
        else:
            model = model.eval()

        return tokenizer, model, device

    except Exception as e:
        st.error("❌ Failed to load model. Check model name or your internet.")
        st.exception(e)
        st.stop()
        return None, None, None


# =========================
# Load Embeddings
# =========================
embeddings = get_embeddings(embedding_model_name)

# =========================
# Process Uploaded Docs
# =========================
raw_docs = load_documents(uploaded_files)

if raw_docs:
    with st.spinner("📄 Processing documents..."):
        chunks = chunk_documents(raw_docs)
        vectorstore = build_or_load_faiss(chunks, embeddings, persist_dir or None)
        st.success(f"✅ Indexed {len(chunks)} chunks from {len(raw_docs)} file(s).")
elif persist_dir and Path(persist_dir).exists():
    with st.spinner("📂 Loading existing FAISS index..."):
        # Dummy chunk to satisfy cache
        dummy_chunks = [{"text": "dummy", "source": "none", "chunk_id": "0"}]
        vectorstore = build_or_load_faiss(dummy_chunks, embeddings, persist_dir)
        st.success(f"📁 Loaded FAISS index from `{persist_dir}`.")
else:
    vectorstore = None
    st.info("📤 Upload PDFs or text files to build a knowledge base.")


# =========================
# Load LLM
# =========================
with st.spinner("📥 Downloading and loading LLM... This may take 1-2 minutes."):
    try:
        tokenizer, model, device = load_local_model(llm_model_name, HF_TOKEN)
        st.success(f"🟢 Model loaded on `{device.upper()}`")
    except Exception as e:
        st.error("❌ Failed to load model. Try a smaller one like 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'")
        st.exception(e)
        st.stop()


# =========================
# Format Prompt for TinyLlama
# =========================
def format_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n".join([f"- {c}" for c in context_chunks]) if context_chunks else "No context provided."
    system_msg = "You are a helpful AI assistant. Use the context to answer the question. If unsure, say 'I don't know.'"
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def generate_answer(prompt: str, temperature: float, top_p: float, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # Extract assistant's reply
    if "<|im_start|>assistant" in full_text:
        try:
            answer = full_text.split("<|im_start|>assistant")[-1]
            answer = answer.split("<|im_end|>")[0].strip()
            return answer
        except Exception:
            pass
    return full_text.strip()


# =========================
# Chat UI
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

prompt = st.text_input("💬 Ask a question about your documents:", value="")
ask = st.button("📤 Ask", type="primary")


def retrieve_chunks(query: str, k: int) -> list[dict]:
    if not vectorstore:
        return []
    docs = vectorstore.similarity_search(query, k=k)
    return [
        {
            "text": d.page_content.strip(),
            "source": d.metadata.get("source", "unknown"),
            "chunk_id": d.metadata.get("chunk_id", "")
        }
        for d in docs
    ]


if ask and prompt.strip():
    with st.spinner("🔍 Retrieving relevant context..."):
        retrieved = retrieve_chunks(prompt, k_retrieval)
        context_snippets = [r["text"] for r in retrieved]

    full_prompt = format_prompt(prompt, context_snippets)

    with st.spinner("🧠 Generating answer..."):
        t0 = time.time()
        answer = generate_answer(full_prompt, temperature, top_p, max_new_tokens)
        latency = time.time() - t0

    st.session_state.history.append({
        "question": prompt,
        "answer": answer,
        "retrieved": retrieved,
        "latency": latency
    })


# =========================
# Display Chat History
# =========================
for turn in reversed(st.session_state.history):
    with st.container(border=True):
        st.markdown(f"**🧑 You:** {turn['question']}")
        st.markdown(f"**🤖 Assistant:** {turn['answer']}")
        st.caption(f"⏱️ {turn['latency']:.2f}s | Retrieved {len(turn['retrieved'])} chunks")
        if show_chunks and turn["retrieved"]:
            with st.expander("🔎 View Retrieved Chunks"):
                for i, r in enumerate(turn["retrieved"], 1):
                    st.markdown(f"**{i}. 📄 `{r['source']}`**")
                    st.write(r["text"][:800] + ("..." if len(r["text"]) > 800 else ""))


# =========================
# Footer
# =========================
st.divider()
st.caption("""
⚡ RAG Chatbot | Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Embeddings: `all-MiniLM-L6-v2`  
Tip: Add `.streamlit/config.toml` with `[server]\nfileWatcherType = "none"` to avoid startup errors.
""")
