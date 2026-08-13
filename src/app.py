import streamlit as st
import rag_pipeline
import session_ingest
import verify_paper

st.set_page_config(
    page_title="Standards Compliance Assistant",
    page_icon=":material/verified:",
    layout="wide",
)

st.markdown("""
<style>
.stApp { max-width: 1000px; margin: 0 auto; }
div[data-testid="stSidebar"] { background-color: #f8f9fb; }
.stButton button {
    background-color: #2563eb; color: white; border-radius: 8px;
    padding: 0.5rem 1.5rem; font-weight: 500; border: none;
}
.stButton button:hover { background-color: #1d4ed8; }
h1 { font-size: 2rem !important; }
div[data-testid="stMetric"] { background-color: #f8f9fb; border-radius: 12px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns([0.06, 0.94])
    with col1:
        st.markdown("### :material/verified:")
    with col2:
        st.title("Self-Verifying Standards Compliance Assistant")

st.caption("Agentic RAG with hybrid retrieval, relevance grading, and groundedness verification")
st.divider()

mode = st.sidebar.segmented_control(
    "Mode",
    options=["Ask", "Upload", "Verify"],
    format_func=lambda x: {
        "Ask": ":material/quiz: Ask built-in doc",
        "Upload": ":material/upload_file: Upload your PDF",
        "Verify": ":material/fact_check: Check a paper",
    }[x],
    default="Ask",
)

if mode is None:
    mode = "Ask"


def show_status(status):
    if status == "Verified":
        st.success(":material/check_circle: Status: Verified", icon=":material/check_circle:")
    else:
        st.warning(f":material/warning: Status: {status}", icon=":material/warning:")


def show_sources(sources, label="Sources"):
    if sources:
        st.markdown(f"**:material/description: {label}**")
        for c in sources:
            with st.expander(f":material/article: Page {c['metadata'].get('page') + 1}"):
                st.write(c["text"])


if mode == "Ask":
    with st.container(border=True):
        st.subheader(":material/quiz: Ask the built-in document")
        st.caption("Currently loaded: *Attention Is All You Need*")
        question = st.text_input(
            "Your question",
            placeholder="e.g. What is multi-head attention?",
            label_visibility="collapsed",
        )
        ask = st.button(":material/search: Get Answer", type="primary")

    if ask and question:
        with st.spinner(":material/search: Retrieving, grading, and verifying...", show_time=True):
            result = rag_pipeline.answer_question(question)

        with st.container(border=True):
            show_status(result["status"])
            st.markdown("### :material/subject: Answer")
            st.write(result["answer"])
            show_sources(result["sources"])

elif mode == "Upload":
    with st.container(border=True):
        st.subheader(":material/upload_file: Upload your own PDF")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], key="qa_upload")

        if uploaded_file is not None:
            if "session_vectorstore" not in st.session_state or st.session_state.get("uploaded_filename") != uploaded_file.name:
                with st.spinner(":material/upload: Reading and indexing your PDF...", show_time=True):
                    vectorstore, num_pages, num_chunks = session_ingest.ingest_uploaded_pdf(uploaded_file)
                    st.session_state["session_vectorstore"] = vectorstore
                    st.session_state["uploaded_filename"] = uploaded_file.name

                c1, c2 = st.columns(2)
                c1.metric("Pages indexed", num_pages)
                c2.metric("Chunks created", num_chunks)

            question = st.text_input(
                "Your question",
                placeholder="Ask something about your uploaded PDF",
                label_visibility="collapsed",
            )
            ask = st.button(":material/search: Get Answer", type="primary", key="upload_ask")

    if uploaded_file is not None and ask and question:
        with st.spinner(":material/search: Retrieving, grading, and verifying...", show_time=True):
            result = rag_pipeline.answer_question_from_store(
                st.session_state["session_vectorstore"], question
            )

        with st.container(border=True):
            show_status(result["status"])
            st.markdown("### :material/subject: Answer")
            st.write(result["answer"])
            show_sources(result["sources"])

else:  # Verify
    with st.container(border=True):
        st.subheader(":material/fact_check: Check a paper's claims")
        st.caption(
            "Each factual claim is checked against the built-in reference "
            "document, and flagged if outdated or unsupported, with a "
            "suggested correction."
        )
        paper_text = st.text_area(
            "Paste paper text here",
            height=180,
            placeholder="e.g. The Transformer model uses a dimension of 256 for its embeddings...",
            label_visibility="collapsed",
        )
        check = st.button(":material/fact_check: Check Claims", type="primary")

    if check and paper_text.strip():
        with st.spinner(":material/fact_check: Extracting and verifying claims - this may take a minute...", show_time=True):
            report = verify_paper.verify_paper_against_builtin(paper_text)

        st.metric("Claims extracted", report["total_claims"])

        for r in report["results"]:
            with st.container(border=True):
                st.markdown(f"**:material/format_quote: Claim:** {r['claim']}")

                if r["verdict"] == "Verified":
                    st.success(f":material/check_circle: {r['verdict']}", icon=":material/check_circle:")
                else:
                    st.warning(f":material/warning: {r['verdict']}", icon=":material/warning:")

                st.write(r["explanation"])

                if r["suggestion"]:
                    st.info(f":material/edit: **Suggested correction:** {r['suggestion']}", icon=":material/edit:")

                show_sources(r["sources"], label="Matched source(s)")

st.divider()
st.caption(":material/build: Built with LangChain, Groq, ChromaDB, and Streamlit — hybrid retrieval + self-correction pipeline")