# streamlit_app.py
import streamlit as st
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# === Set Page Config ===
st.set_page_config(page_title="Fusion RAG Chatbot", layout="wide")
st.title("⚡ Fusion RAG Chatbot (Local LLM + FAISS)")

# === Configuration (Modify these paths/tokens as needed) ===
FAISS_FOLDER = "content/faiss_index"  # Folder containing faiss.index, index.pkl, etc.
EMBED_MODEL = "sentence-transformers/sentence-t5-large"
LLM_MODEL = "meta-llama/Llama-2-7b-chat-hf"
HF_TOKEN = "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"  # Replace with your token

# Optional: Set environment variable (if not already set)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN


# === Load Embedding Model & FAISS Vectorstore (Cached) ===
@st.cache_resource
def load_vectorstore():
    if not os.path.exists(FAISS_FOLDER):
        st.error(f"❌ FAISS index folder not found: {FAISS_FOLDER}")
        st.info("Please ensure your FAISS files (index.faiss, index.pkl) are in the `content/faiss_index/` directory.")
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
            allow_dangerous_deserialization=True  # Required for loading serialized objects
        )
        st.success("✅ FAISS vectorstore loaded!")
        return db.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"❌ Failed to load FAISS index: {e}")
        return None


# === Load LLM and Tokenizer (Cached) ===
@st.cache_resource
def load_llm():
    try:
        st.info(f"🦙 Loading LLM: {LLM_MODEL}... (this may take a minute)")

        tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL,
            use_auth_token=HF_TOKEN,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            use_auth_token=HF_TOKEN,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",  # Automatically uses GPU if available
            trust_remote_code=True,
            resume_download=True
        )

        st.success("✅ LLM loaded successfully!")
        return tokenizer, model

    except Exception as e:
        st.error(f"❌ Failed to load LLM. Common issues:")
        st.markdown(f"""
        - Make sure you have access to **{LLM_MODEL}** on Hugging Face
        - Your token is correct and has **gated model access**
        - Run: `huggingface-cli login --token={HF_TOKEN}` locally if needed
        - Error: `{e}`
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
You are a helpful AI assistant. Answer the user's question using **only** the context below.

### Instructions:
- Do NOT use prior knowledge.
- If the context doesn't contain the answer, say: "The context does not provide this information."
- Be concise and clear.
- Use bullet points or short paragraphs.

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

        if not answer or "context does not provide" in answer.lower():
            return "The context does not provide this information."
        return answer
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

# === Chat Interface ===
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question...")

if user_input and retriever and tokenizer and model:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = fusion_rag_answer(user_input, retriever, tokenizer, model)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# === Sidebar Info ===
with st.sidebar:
    st.header("📚 About")
    st.markdown("""
    This is a **Fusion RAG Chatbot** that:
    - Uses **local FAISS** vector store
    - Loads **Llama-2-7b-chat-hf** locally
    - Applies **query expansion + RRF**
    - Runs fully offline after download

    🔐 Your data never leaves this machine.
    """)

    st.subheader("📁 FAISS Index")
    if os.path.exists(FAISS_FOLDER):
        st.success("Index found")
    else:
        st.error("Index missing")

    st.subheader("🦙 LLM Status")
    if model:
        device = model.device
        st.success(f"Loaded on {device}")
    else:
        st.warning("LLM not loaded")
