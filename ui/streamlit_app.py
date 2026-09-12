import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import streamlit as st

from app.rag.pipeline import RAGPipeline
from app.memory.conversation import ConversationMemory


st.set_page_config(
    page_title="Granimals Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
)


@st.cache_resource
def load_rag_pipeline():
    return RAGPipeline()


@st.cache_resource
def load_conversation_memory():
    return ConversationMemory()


rag_pipeline = load_rag_pipeline()
conversation_memory = load_conversation_memory()


st.title("🤖 Granimals Knowledge Assistant")
st.caption(
    "Ask questions about your uploaded knowledge documents."
)


# ---------------------------------------------------------
# Multi-PDF Upload
# ---------------------------------------------------------

st.sidebar.header("📚 Document Upload")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:

    if st.sidebar.button(
        "Process Documents"
    ):

        documents_dir = Path(
            "data/documents"
        )

        documents_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        progress_text = st.sidebar.empty()

        for index, uploaded_file in enumerate(
            uploaded_files,
            start=1,
        ):

            progress_text.text(
                f"Processing {index}/{len(uploaded_files)}: "
                f"{uploaded_file.name}"
            )

            safe_filename = Path(
                uploaded_file.name
            ).name

            pdf_path = (
                documents_dir
                / safe_filename
            )

            pdf_path.write_bytes(
                uploaded_file.getbuffer()
            )

            try:

                result = (
                    rag_pipeline.ingest_document(
                        pdf_path
                    )
                )

                st.sidebar.success(
                    f"✓ {uploaded_file.name} processed"
                )

            except Exception as exc:

                st.sidebar.error(
                    f"✗ Failed: {uploaded_file.name}"
                )

                st.sidebar.exception(
                    exc
                )

        progress_text.text(
            "Document processing completed."
        )


# ---------------------------------------------------------
# Conversation ID
# ---------------------------------------------------------

if "conversation_id" not in st.session_state:

    import uuid

    st.session_state.conversation_id = str(
        uuid.uuid4()
    )


conversation_id = st.session_state.conversation_id


# ---------------------------------------------------------
# Chat history
# ---------------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message.get("sources"):

            with st.expander(
                "Sources"
            ):

                for source in message[
                    "sources"
                ]:

                    document = source.get(
                        "document",
                        "Unknown document",
                    )

                    page = source.get(
                        "page",
                        "Unknown",
                    )

                    content_type = source.get(
                        "content_type",
                        "text",
                    )

                    st.write(
                        f"📄 {document} | "
                        f"Page {page} | "
                        f"{content_type}"
                    )


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    # Display user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # Get previous conversation history

    history = (
        conversation_memory.get_messages(
            conversation_id
        )
    )


    # Generate answer

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            result = rag_pipeline.answer(
                question,
                conversation_history=history,
            )


        answer = result.get(
            "answer",
            "The information was not found.",
        )

        sources = result.get(
            "sources",
            [],
        )

        confidence = result.get(
            "confidence",
            0.0,
        )


        st.markdown(answer)


        # Confidence

        st.caption(
            f"Confidence: {confidence:.2f}"
        )


        # Sources

        if sources:

            with st.expander(
                "Sources"
            ):

                for source in sources:

                    document = source.get(
                        "document",
                        "Unknown document",
                    )

                    page = source.get(
                        "page",
                        "Unknown",
                    )

                    content_type = source.get(
                        "content_type",
                        "text",
                    )

                    st.write(
                        f"📄 {document} | "
                        f"Page {page} | "
                        f"{content_type}"
                    )


    # Store conversation

    conversation_memory.add_user_message(
        conversation_id,
        question,
    )

    conversation_memory.add_ai_message(
        conversation_id,
        answer,
    )


    # Store UI history

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )