import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
import re

load_dotenv()

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MAX_CITATION_SNIPPET_LENGTH = 180
# CHUNK_SIZE/CHUNK_OVERLAP balance QA context continuity with retrieval precision.

# --- HELPER FUNCTIONS ---

def get_video_id(url):
    """Extracts the video ID from a YouTube URL."""
    # Regex to find video ID in various YouTube URL formats
    match = re.search(r"(?<=v=)[^&#]+|(?<=be/)[^&#]+|(?<=embed/)[^&#]+|(?<=shorts/)[^&#]+", url)
    if match:
        return match.group(0)
    # Fallback for googleusercontent URLs
    if 'googleusercontent.com/youtube.com' in url:
        return url.split('/')[-1].split('?')[0]
    return None

def format_timestamp(seconds):
    """Converts float seconds to MM:SS format."""
    total_seconds = max(0, int(seconds))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def get_transcript_segments(url):
    """
    Fetches transcript segments for a YouTube video.
    """
    try:
        video_id = get_video_id(url)
        if not video_id:
            return None, "Could not extract video ID from the URL."

        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id)

        segments = []
        for item in transcript_list:
            text = getattr(item, "text", "").strip()
            if not text:
                continue
            start = float(getattr(item, "start", 0.0))
            segments.append({"text": text, "start": start})

        if not segments:
            return None, "Transcript was fetched but empty."

        return segments, None
    except Exception as e:
        return None, f"An error occurred while fetching the transcript: {e}"


def get_vector_store(transcript_segments):
    """
    Creates and returns a FAISS vector store from transcript segments.
    """
    if not transcript_segments:
        return None

    documents = [
        Document(
            page_content=segment["text"],
            metadata={
                "timestamp": format_timestamp(segment["start"])
            }
        )
        for segment in transcript_segments
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)

    emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, emb_model)

    return vectorstore


def build_citations(docs, limit=3):
    """Builds concise citation entries from retrieved documents."""
    citations = []
    seen = set()
    for doc in docs:
        timestamp = doc.metadata.get("timestamp", "00:00")
        snippet = " ".join(doc.page_content.split())
        if timestamp in seen:
            continue
        seen.add(timestamp)
        truncated_snippet = snippet[:MAX_CITATION_SNIPPET_LENGTH]
        if len(snippet) > MAX_CITATION_SNIPPET_LENGTH:
            truncated_snippet += "..."
        citations.append({"timestamp": timestamp, "snippet": truncated_snippet})
        if len(citations) >= limit:
            break
    return citations


def create_rag_chain(vectorstore, gemini_api_key):
    """
    Creates and returns an invoke function for grounded RAG responses.
    """
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=gemini_api_key)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6}
    )

    prompt = PromptTemplate(
        template="""
        You are an AI assistant specialized in answering questions strictly from provided video transcript context.

        Instructions:
        - Use only the supplied context
        - Do not hallucinate
        - If information is unavailable, clearly say so
        - Provide concise and accurate answers
        - Summarize long explanations clearly

        User question:
        {query}

        Context:
        {context}
        """,
        input_variables=['query', 'context']
    )

    qa_chain = prompt | llm | StrOutputParser()

    def invoke_rag(query):
        docs = retriever.invoke(query)
        context = format_docs(docs)
        answer = qa_chain.invoke({"query": query, "context": context})
        return {"answer": answer, "citations": build_citations(docs)}

    return invoke_rag


def format_docs(docs):
    """Combines document contents into a transcript context string."""
    return "\n\n".join(
        f"[{doc.metadata.get('timestamp', '00:00')}] {doc.page_content}"
        for doc in docs
    )

# --- STREAMLIT UI ---

st.set_page_config(page_title="TranscriptIQ", page_icon="🎬", layout="wide")

st.title("🎬 TranscriptIQ")
st.markdown("AI-powered YouTube transcript Q&A with retrieval-augmented generation.")

# --- SIDEBAR for Inputs ---
with st.sidebar:
    st.header("Setup")

    env_api_key = os.getenv("GOOGLE_API_KEY", "")
    youtube_url = st.text_input("Enter YouTube URL:", key="youtube_url_input")
    gemini_api_key = st.text_input(
        "Enter Gemini API Key (optional if set in .env):",
        type="password",
        key="gemini_api_key_input",
        value=env_api_key
    )

    if st.button("Process Video", key="process_button"):
        active_api_key = gemini_api_key.strip() or env_api_key.strip()
        if not youtube_url:
            st.error("Please enter a YouTube URL.")
        elif not active_api_key:
            st.error("Please set GOOGLE_API_KEY in .env or provide it in the sidebar.")
        else:
            with st.spinner("Processing video... This may take a moment."):
                os.environ["GOOGLE_API_KEY"] = active_api_key

                transcript_segments, error_message = get_transcript_segments(youtube_url)
                if error_message:
                    st.error(error_message)
                    st.session_state.rag_chain = None
                else:
                    vector_store = get_vector_store(transcript_segments)
                    if vector_store:
                        st.session_state.rag_chain = create_rag_chain(vector_store, active_api_key)
                        st.success("Video processed successfully! You can now ask questions.")
                        st.session_state.messages = []
                    else:
                        st.error("Could not create vector store from the transcript.")
                        st.session_state.rag_chain = None

# --- CHAT INTERFACE ---

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("citations"):
            with st.expander("Sources"):
                for citation in message["citations"]:
                    st.markdown(f"- **{citation['timestamp']}** — {citation['snippet']}")

# Handle new user input
if prompt := st.chat_input("Ask a question about the video..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "rag_chain" not in st.session_state or st.session_state.rag_chain is None:
        st.warning("Please process a video first using the sidebar.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_payload = st.session_state.rag_chain(prompt)
                response_text = response_payload["answer"]
                citations = response_payload["citations"]
                st.markdown(response_text)
                if citations:
                    with st.expander("Sources"):
                        for citation in citations:
                            st.markdown(f"- **{citation['timestamp']}** — {citation['snippet']}")

        st.session_state.messages.append(
            {"role": "assistant", "content": response_text, "citations": citations}
        )
