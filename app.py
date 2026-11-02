import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

DATA_DIR = "./data"
DB_DIR = "./chroma_db"
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="Document Chatbot", page_icon="📚")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing! Check your .env file.")
    st.stop()



# ---------- Cached resources (created once, reused on every rerun) ----------
@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


@st.cache_resource
def get_rag_chain():
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might "
                   "reference context in the chat history, formulate a standalone "
                   "question which can be understood without the chat history. "
                   "Do NOT answer the question, just reformulate it if needed, "
                   "otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based ONLY on the provided context below. "
                   "If the answer cannot be found in the context, say "
                   "\"I don't have enough information in the provided context.\"\n\n"
                   "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain)


# ---------- Indexing helpers ----------
def indexed_sources():
    data = get_vectorstore().get(include=["metadatas"])
    return {m.get("source") for m in data["metadatas"] if m}


def index_file(path):
    """Index one file unless it's already in the database. Returns number of chunks added."""
    if path in indexed_sources():
        return 0
    loader = PyPDFLoader(path) if path.lower().endswith(".pdf") else TextLoader(path, encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(loader.load())
    if chunks:
        get_vectorstore().add_documents(chunks)
    return len(chunks)


# ---------- Sidebar: upload + indexing ----------
with st.sidebar:
    st.header("📁 Documents")
    uploads = st.file_uploader("Upload .txt or .pdf files", type=["txt", "pdf"], accept_multiple_files=True)

    if uploads and st.button("Add to knowledge base"):
        with st.spinner("Indexing..."):
            for f in uploads:
                path = os.path.join(DATA_DIR, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                added = index_file(path)
                if added:
                    st.success(f"{f.name}: {added} chunks added")
                else:
                    st.info(f"{f.name}: already indexed")

    if st.button("Index everything in ./data"):
        with st.spinner("Indexing..."):
            total = 0
            for name in os.listdir(DATA_DIR):
                if name.lower().endswith((".txt", ".pdf")):
                    total += index_file(os.path.join(DATA_DIR, name))
            st.success(f"{total} new chunks added")

    st.divider()
    st.caption("Indexed files:")
    for s in sorted(s for s in indexed_sources() if s):
        st.write("•", os.path.basename(s))

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()


# ---------- Chat state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []      # what we display
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # what we send to the chain


def render_sources(docs):
    with st.expander("📄 Sources"):
        for d in docs:
            name = os.path.basename(d.metadata.get("source", "unknown"))
            page = d.metadata.get("page")
            label = f"**{name}**" + (f" (page {page + 1})" if page is not None else "")
            st.markdown(label)
            st.caption(d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""))


st.title("📚 Chat with your documents")

# Replay earlier messages (Streamlit reruns the script on every interaction)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# ---------- Handle new question ----------
if query := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    rag_chain = get_rag_chain()
    sources = []

    def answer_stream():
        for chunk in rag_chain.stream({
            "input": query,
            "chat_history": st.session_state.chat_history[-6:],
        }):
            if "context" in chunk:
                sources.extend(chunk["context"])
            if "answer" in chunk:
                yield chunk["answer"]

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(answer_stream())
            if sources:
                render_sources(sources)
        except Exception as e:
            answer = f"Something went wrong: {e}"
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.session_state.chat_history.append(HumanMessage(content=query))
    st.session_state.chat_history.append(AIMessage(content=answer))