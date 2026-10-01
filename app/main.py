from app.qa_engine import generate_rag_response
from app.vector_store import add_chunks_to_store
from app.embeddings import generate_embeddings
from app.pdf_processor import process_pdf_file
import streamlit as st
import os
import sys
import tempfile
import base64

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


st.set_page_config(
    page_title="NimBot | AI Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS (Candy Cyan & White) =================
st.markdown("""
<style>
    /* Force main background white */
    .stApp { background-color: #FFFFFF !important; }

    /* Sidebar logo styling */
    .sidebar-logo { display: block; margin: 10px auto 5px auto; width: 150px; }
    .sidebar-title { text-align: center; font-size: 0.85rem; font-weight: 700; color: #0099CC !important; margin-bottom: 20px; }

    /* Social links */
    .social-link {
        display: flex; align-items: center; gap: 8px; padding: 6px 0;
        text-decoration: none !important; color: #1A1A1A !important; font-weight: 600; font-size: 0.95rem;
    }
    .social-link:hover { color: #00D4FF !important; }
    .social-icon { width: 20px; height: 20px; }

    /* File Uploader Box (Candy Cyan with White text) */
    [data-testid="stFileUploadDropzone"] {
        background-color: #00D4FF !important;
        border: 2px dashed #0099CC !important;
        border-radius: 12px;
    }
    [data-testid="stFileUploadDropzone"] * {
        color: #FFFFFF !important;
        font-weight: 600;
    }

    /* Chat Input Box (Light Cyan) */
    div[data-testid="stChatInput"] {
        background-color: #E0F7FA !important;
        border: 2px solid #00D4FF !important;
        border-radius: 15px !important;
    }
    div[data-testid="stChatInput"] textarea {
        color: #1A1A1A !important;
        font-weight: 500;
    }

    /* Source badge */
    .source-badge {
        background-color: #E0F7FA;
        color: #0083B0 !important;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
        margin-top: 10px;
        border: 1px solid #00D4FF;
    }
    
    /* Main Chat Bubbles */
    div[data-testid="stChatMessage"] {
        background-color: #F8FDFF;
        border: 1px solid #B2EBF2;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


def get_base64_image(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""


# ================= SIDEBAR =================
logo_path = os.path.join(PROJECT_ROOT, "assets", "nimbot_logo.png")
logo_b64 = get_base64_image(logo_path)

if logo_b64:
    st.sidebar.markdown(
        f'<img src="data:image/png;base64,{logo_b64}" class="sidebar-logo">', unsafe_allow_html=True)

st.sidebar.markdown(
    '<div class="sidebar-title">Chat with your PDFs — Zero Hallucinations</div>', unsafe_allow_html=True)

# GitHub & LinkedIn with icons
github_icon = "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg"
linkedin_icon = "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg"

st.sidebar.markdown(f"""
<a href="https://github.com/nimasharifiniko" target="_blank" class="social-link">
    <img src="{github_icon}" class="social-icon"> GitHub
</a>
<a href="https://linkedin.com/in/nimasharifiniko" target="_blank" class="social-link">
    <img src="{linkedin_icon}" class="social-icon"> LinkedIn
</a>
""", unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.subheader("📂 Upload Document")
uploaded_file = st.sidebar.file_uploader("Drop your PDF here", type=["pdf"])

# ================= MAIN AREA =================
st.title("💬 Chat with your Document")

# Session state setup
if "messages" not in st.session_state:
    # Default Welcome Message (Fixes the empty gap!)
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! 👋 I am NimBot. Upload your PDF document in the sidebar, and I'll answer any questions you have with exact page citations.", "sources": []}
    ]
if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None

# Process PDF
if uploaded_file is not None:
    if st.session_state.indexed_file != uploaded_file.name:
        with st.spinner("Reading PDF and building vector index..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            chunks = process_pdf_file(tmp_path)
            if chunks:
                texts = [c["text"] for c in chunks]
                embeddings = generate_embeddings(texts)
                add_chunks_to_store(chunks, embeddings)
                st.session_state.indexed_file = uploaded_file.name

                # Reset chat and add success message
                st.session_state.messages = [
                    {"role": "assistant", "content": f"✅ Successfully processed **{uploaded_file.name}**! What would you like to know about it?", "sources": []}
                ]
            else:
                st.sidebar.error("Failed to extract text from PDF.")
            os.remove(tmp_path)

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            pages = ", ".join(f"Page {p}" for p in msg["sources"])
            st.markdown(
                f'<div class="source-badge">📍 Source: {pages}</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask NimBot about the document..."):
    if not st.session_state.indexed_file:
        st.warning("⚠️ Please upload a PDF in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                res = generate_rag_response(prompt, top_k=3)
                st.markdown(res["answer"])
                if res["sources"]:
                    pages = ", ".join(f"Page {p}" for p in res["sources"])
                    st.markdown(
                        f'<div class="source-badge">📍 Source: {pages}</div>', unsafe_allow_html=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": res["answer"],
                    "sources": res["sources"]
                })
