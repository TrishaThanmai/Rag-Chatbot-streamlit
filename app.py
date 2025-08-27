import streamlit as st
from pathlib import Path
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ========== Streamlit Config ==========
st.set_page_config(page_title="Fusion RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Fusion RAG Chatbot")

# ========== Config ==========
HF_TOKEN = st.secrets["HF_TOKEN"]   # Hugging Face token stored in Streamlit secrets
APP_DIR = Path(__file__).resolve().parent
FAISS_DIR = APP_DIR / "faiss_index"
INDEX_NAME = "index"

EMBED_MODEL = "sentence-transformers/sentence-t5-large"
LLM_MODEL = "google/gemma-2-9b"   # or meta-llama/Llama-3.2-3b-instruct

# ========== Cache Helpers ==========
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})

def ensure_faiss(emb):
    """Create FAISS index if missing, else load existing one."""
    if not FAISS_DIR.exists():
        st.warning("FAISS index not found, creating a new one...")
        # Temporary fallback docs (replace with your real dataset later)
        texts = [
            "Fusion RAG retrieves documents with multiple queries.",
            "Reciprocal Rank Fusion merges results for better accuracy.",
            "This is a fallback document when no FAISS index is found."
        ]
        db = FAISS.from_texts(texts, emb)
        db.save_local(str(FAISS_DIR), index_name=INDEX_NAME)

    return FAISS.load_local(
        folder_path=str(FAISS_DIR),
        embeddings=emb,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )

@st.cache_resource
def load_faiss(_emb):
    return ensure_faiss(_emb)

@st.cache_resource
def get_hf_client():
    return InferenceClient(model=LLM_MODEL, token=HF_TOKEN)

# Load resources
embeddings = load_embeddings()
db = load_faiss(embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})
client = get_hf_client()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ========== Fusion RAG Utils ==========
def generate_queries(original_query: str):
    return [
        original_query,
        f"Explain in detail: {original_query}",
        f"What are the benefits of {original_query}?",
        f"What are the challenges or drawbacks of {original_query}?",
        f"Give a real-world application of {original_query}"
    ]

def reciprocal_rank_fusion(results_dict, k=60):
    """Fuse multiple retrieval results using RRF."""
    fused_scores = {}
    for q, docs in results_dict.items():
        for rank, doc in enumerate(docs):
            fused_scores[doc.page_content] = fused_scores.get(doc.page_content, 0) + 1 / (rank + k)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

def fusion_rag_answer(query: str):
    # 1️ Generate multiple queries
    expanded_queries = generate_queries(query)

    # 2️ Retrieve documents for each query
    all_results = {q: retriever.get_relevant_documents(q) for q in expanded_queries}

    # 3️ Fuse results
    reranked = reciprocal_rank_fusion(all_results)

    # 4️ Build context from top docs
    top_passages = [doc for doc, _ in reranked[:5]]
    context = "\n\n".join(top_passages)

    # 5️ Create prompt
    prompt = f"""
Imagine you are chatting with me as my study buddy. 
I’ll give you some context, and you need to answer my question based on it.  

Here’s how I’d like you to reply:
- Stick only to the details from the context. 
- If the context doesn’t cover it, just say: 
  "The context does not provide this information."
- Write in a friendly, easy-to-follow way. 
- Feel free to break things into short bullets if it helps."

Context:
{context}

Question: {query}

Final Answer:
"""

    # 6️ Query Hugging Face Inference
    response = client.text_generation(
        prompt,
        max_new_tokens=350,
        temperature=0.2,
        do_sample=False,
        stream=False,
    )

    # 7️ Extract Answer
    raw_answer = ""
    if isinstance(response, str):
        raw_answer = response
    elif isinstance(response, dict) and "generated_text" in response:
        raw_answer = response["generated_text"]
    elif isinstance(response, list) and "generated_text" in response[0]:
        raw_answer = response[0]["generated_text"]

    return raw_answer.split("Final Answer:", 1)[-1].strip()

# ========== UI ==========
query = st.text_input("Ask me something:")

if query:
    with st.spinner("Thinking..."):
        answer = fusion_rag_answer(query)
        st.session_state.chat_history.append({"question": query, "answer": answer})

# Display Chat History
if st.session_state.chat_history:
    st.subheader("🤖 Conversation History")
    for i, chat in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}:** {chat['question']}")
        st.markdown(f"**A{i}:** {chat['answer']}")
        st.markdown("---")