# app.py
import os
from pathlib import Path
import time
import io

import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    st.warning("No Hugging Face token found. Set `HF_TOKEN` in Streamlit secrets or env. "
               "Gated models like Gemma will fail to load.")
    st.stop()

# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    st.header("⚙️ Settings")

    # Embedding model
    embedding_model_name = st.text_input(
        "Embedding model",
        value="sentence-transformers/all-MiniLM-L6-v2",
        help="Used for FAISS vector search."
    )

    # LLM model - now set to Gemma 2 9B
    llm_model_name = st.text_input(
        "LLM model (Transformers)",
        value="google/gemma-2-9b",
        help="Hugging Face model ID. Must have access via HF_TOKEN. Example: 'google/gemma-2-9b'"
    )

    st.caption("💡 Tip: For lower VRAM, try `google/gemma-2-2b` or enable 4-bit.")

    temperature = st.slider("Temperature", 0.0, 1.5, 0.2, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.01)
    max_new_tokens = st.slider("Max new tokens", 16, 1024, 256, 16)
    k_retrieval = st.slider("Top-k chunks", 1, 10, 4, 1)

    use_gpu = st.checkbox("Use GPU if available", value=True)
    use_4bit = st.checkbox("Use 4-bit quantization (save VRAM)", value=True)
    show_chunks = st.checkbox("Show retrieved chunks", value=False)

    st.divider()
    st.subheader("📄 Load Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDFs or text files", type=["pdf", "txt"], accept_multiple_files=True
    )
    persist_dir = st.text_input(
        "FAISS persist directory (optional)",
        value="faiss_index",
        help="If set, FAISS index will be saved/loaded here."
    )
    rebuild_index = st.checkbox("Rebuild index on upload", value=False)

# =========================
# Utilities
# =========================
@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}  # Embeddings are lightweight, CPU is fine
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
    for encoding in ["utf-8", "latin-1"]:
        try:
            return file_bytes.decode(encoding)
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

def chunk_documents(raw_docs: list[dict], chunk_size=800, chunk_overlap=120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
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
            st.info("Building new FAISS index...")
            vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
            path.mkdir(parents=True, exist_ok=True)
            vs.save_local(str(path))
            st.success(f"Saved FAISS index to {path}")
        else:
            st.info(f"Loading FAISS index from {path}...")
            vs = FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    else:
        vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    return vs

# =========================
# Load Model - Gemma-2-9b Compatible
# =========================
@st.cache_resource(show_spinner="Loading LLM...")
def load_local_model(model_name: str, token: str, prefer_gpu: bool, use_4bit: bool):
    try:
        from transformers import AutoConfig
        # Validate access
        AutoConfig.from_pretrained(model_name, token=token)
    except Exception as e:
        st.error("Failed to access model. Did you accept the license at https://huggingface.co/google/gemma-2-9b?")
        st.exception(e)
        st.stop()

    device = "cuda" if (prefer_gpu and torch.cuda.is_available()) else "cpu"
    dtype = torch.float16 if (device == "cuda") else torch.float32
    device_map = "auto" if device == "cuda" else None

    # Quantization config
    bnb_config = None
    if use_4bit and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        dtype = torch.float16

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set chat template for Gemma
    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<start_of_turn>' + message['role'] + '\n' + message['content'].strip() + '<end_of_turn>\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<start_of_turn>model\n' }}"
            "{% endif %}"
        )

    # Model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            torch_dtype=dtype,
            device_map=device_map,
            quantization_config=bnb_config,
            trust_remote_code=False,
            revision="main"
        )
    except Exception as e:
        st.error("Failed to load model. Check VRAM or disable 4-bit.")
        st.exception(e)
        st.stop()
        return None, None, None

    return tokenizer, model, device


# Load embeddings
embeddings = get_embeddings(embedding_model_name)

# =========================
# Process Uploaded Docs
# =========================
raw_docs = load_documents(uploaded_files)

if raw_docs:
    with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
        chunks = chunk_documents(raw_docs)
        vectorstore = build_or_load_faiss(chunks, embeddings, persist_dir or None)
        st.success(f"✅ Indexed {len(chunks)} chunks from {len(raw_docs)} file(s).")
elif persist_dir and Path(persist_dir).exists():
    with st.spinner("Loading existing FAISS index..."):
        dummy_chunks = [{"text": "dummy", "source": "none", "chunk_id": "0"}]
        vectorstore = build_or_load_faiss(dummy_chunks, embeddings, persist_dir)
        st.success(f"📁 Loaded FAISS index from `{persist_dir}`.")
else:
    vectorstore = None
    st.info("📤 Upload PDFs or text files to build a knowledge base.")

# =========================
# Load LLM
# =========================
with st.spinner("Downloading and loading LLM... This may take a few minutes."):
    try:
        tokenizer, model, device = load_local_model(llm_model_name, HF_TOKEN, use_gpu, use_4bit)
        if model is None:
            st.stop()
        st.success(f"🟢 Model loaded on `{device.upper()}` | 4-bit: `{use_4bit}`")
    except Exception as e:
        st.error("❌ Failed to load model. Check model name, token, or hardware.")
        st.exception(e)
        st.stop()

# =========================
# Prompt Formatting for Gemma
# =========================
def format_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join([f"- {c}" for c in context_chunks]) if context_chunks else "No context available."
    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    messages = [{"role": "user", "content": user_message}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt

def generate_answer(prompt: str, temperature: float, top_p: float, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # Extract only model response after last <start_of_turn>model
    if "<start_of_turn>model" in full_text:
        try:
            answer = full_text.split("<start_of_turn>model")[-1].strip()
            # Remove <end_of_turn> and anything after
            answer = answer.split("<end_of_turn>")[0].strip()
            return answer
        except Exception:
            pass
    return full_text.strip()

# =========================
# Chat UI
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

prompt = st.text_input("Ask a question about your uploaded documents:", value="")
ask = st.button("Ask", type="primary")

def retrieve_chunks(query: str, k: int) -> list[dict]:
    if not vectorstore:
        return []
    docs = vectorstore.similarity_search(query, k=k)
    out = []
    for d in docs:
        meta = d.metadata or {}
        out.append({
            "text": d.page_content.strip(),
            "source": meta.get("source", "unknown"),
            "chunk_id": meta.get("chunk_id", "")
        })
    return out

if ask and prompt.strip():
    with st.spinner("🔍 Retrieving context..."):
        retrieved = retrieve_chunks(prompt, k_retrieval)
        context_snippets = [r["text"] for r in retrieved]

    full_prompt = format_prompt(prompt, context_snippets)

    with st.spinner("🧠 Generating answer..."):
        t0 = time.time()
        answer = generate_answer(full_prompt, temperature, top_p, max_new_tokens)
        latency = time.time() - t0

    # Save to history
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
        st.caption(f"⏱️ {turn['latency']:.2f}s | Top-{k_retrieval} chunks")
        if show_chunks and turn["retrieved"]:
            with st.expander("🔎 Retrieved Context Chunks"):
                for i, r in enumerate(turn["retrieved"], 1):
                    st.markdown(f"**{i}. 📄 `{r['source']}` · `{r['chunk_id']}`**")
                    st.write(r["text"][:1200] + ("..." if len(r["text"]) > 1200 else ""))

# Footer
st.divider()
st.caption(
    "⚡ Fusion RAG Chatbot | Model: `google/gemma-2-9b` | "
    "RAG flow: Upload → Chunk → FAISS → Retrieve → Prompt → Local LLM"
)
