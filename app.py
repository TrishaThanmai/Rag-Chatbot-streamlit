# streamlit_app.py
import streamlit as st
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# === Set Page Config ===
st.set_page_config(page_title="Fusion RAG Chatbot", layout="wide")
st.title("⚡ Fusion RAG Chatbot (Local Gemma + FAISS)")

# === 🔧 Configuration - Update These Paths/Tokens ===
# ✅ Update this to your actual FAISS folder path
FAISS_FOLDER = r"C:\Users\trish\OneDrive\Desktop\Fusion Rag Streamlit\faiss_index"

# Use a fully open model: Google's Gemma
LLM_MODEL = "google/gemma-2b-it"  # Alternatives: "google/gemma-7b-it" (needs >10GB GPU VRAM)

# Optional: Set your Hugging Face token here or in environment variable
HF_TOKEN = "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"  # Replace with your actual token

# Embedding model used when creating FAISS
EMBED_MODEL = "sentence-transformers/sentence-t5-large"


# === Load Embedding Model & FAISS Vectorstore (Cached) ===
@st.cache_resource
def load_vectorstore():
    if not os.path.exists(FAISS_FOLDER):
        st.error(f"❌ FAISS index folder not found: {FAISS_FOLDER}")
        st.info("💡 Make sure your FAISS files (index.faiss, index.pkl) are in the correct path.")
        return None

    try:
        st.info(f"🧠 Loading embedding model: {EMBED_MODEL}")
        embedding = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        db = FAISS.load_local(
            FAISS_FOLDER,
            embedding,
            allow_dangerous_deserialization=True
        )
        st.success("✅ FAISS vectorstore loaded!")
        return db.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"❌ Failed to load FAISS index: {e}")
        st.exception(e)
        return None


# === Load LLM and Tokenizer (Cached) ===
@st.cache_resource
def load_llm():
    try:
        st.info(f"🟢 Loading LLM: {LLM_MODEL}... (this may take a minute)")

        tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL,
            use_auth_token=HF_TOKEN,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            use_auth_token=HF_TOKEN,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            resume_download=True
        )

        st.success("✅ LLM loaded successfully!")
        return tokenizer, model

    except Exception as e:
        st.error("❌ Failed to load LLM. Common issues:")
        st.markdown(f"""
        - Is your internet connection working?
        - Did you set a valid `HF_TOKEN`?
        - Try running:  
          `huggingface-cli login --token={HF_TOKEN}` in terminal
        - Error details: `{e}`
        """)
        return None, None


# === Fusion RAG Functions ===
def generate_queries(original_query: str):
    """Expand query into multiple perspectives."""
    return [
        original_query,
        f"Explain in detail: {original_query}",
        f"What are the advantages of {original_query}?",
        f"What are the challenges or limitations of {original_query}?",
        f"Give a real-world application of {original_query}"
    ]


def reciprocal_rank_fusion(results_dict, k=60):
    """Fuse results using Reciprocal Rank Fusion."""
    fused_scores = {}
    for query, docs in results_dict.items():
        for rank, doc in enumerate(docs):
            content = doc.page_content.strip()
            if content:
                fused_scores[content] = fused_scores.get(content, 0) + 1 / (rank + k)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


def fusion_rag_answer(query, retriever, tokenizer, model):
    if not query or not query.strip():
        return "Please ask a valid question."

    query = query.strip()

    # Step 1: Expand queries
    expanded_queries = generate_queries(query)
    all_results = {}

    # Step 2: Retrieve for each query
    with st.spinner("🔍 Retrieving relevant documents..."):
        for q in expanded_queries:
            try:
                docs = retriever.get_relevant_documents(q)
                all_results[q] = docs
            except Exception as e:
                st.warning(f"Retrieval failed for '{q}': {e}")

    # Step 3: RRF fusion
    reranked = reciprocal_rank_fusion(all_results)
    if not reranked:
        return "⚠️ No relevant documents retrieved."

    top_passages = [content for content, _ in reranked[:5]]
    context = "\n\n---\n\n".join(top_passages)

    # Step 4: Build prompt
    prompt = f"""
You are a helpful AI assistant. Answer based **only** on the context below.

### Instructions:
- Do NOT use prior knowledge.
- If the answer isn't in the context, say: "The context does not provide this information."
- Be concise and structured.

### Context:
{context}

### Question:
{query}

### Answer:
"""

    # Step 5: Generate with local LLM
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with st.spinner("🧠 Generating answer..."):
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                truncation=True
            )
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the answer part after "Answer:"
        answer_start = full_output.find("### Answer:") + len("### Answer:")
        answer = full_output[answer_start:].strip()

        return answer if answer else "The context does not provide this information."
    except Exception as e:
        return f"❌ Error during generation: {str(e)}"


# === Load Resources ===
retriever = load_vectorstore()
tokenizer, model = load_llm()

# === Initialize Chat History ===
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything based on your knowledge base."}
    ]

# === Display Chat History ===
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# === User Input ===
user_input = st.chat_input("Ask a question...")

if user_input and retriever and tokenizer and model:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and stream assistant response
    with st.chat_message("assistant"):
        response = fusion_rag_answer(user_input, retriever, tokenizer, model)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})


# === Sidebar Info ===
with st.sidebar:
    st.header("📚 About")
    st.markdown("""
    This is a **Fusion RAG Chatbot** that:
    - Uses **local FAISS** vector store
    - Runs **Google's Gemma-2b** locally
    - Applies **query expansion + RRF**
    - No data leaves your machine
    """)

    st.subheader("📁 FAISS Index")
    if os.path.exists(FAISS_FOLDER):
        st.success("✅ Index found")
        st.code(FAISS_FOLDER)
    else:
        st.error("❌ Index not found")

    st.subheader("🟢 LLM Status")
    if model:
        st.success(f"Loaded: {LLM_MODEL}")
        st.write(f"Device: {model.device}")
    else:
        st.warning("Not loaded")
