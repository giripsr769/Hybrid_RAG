import streamlit as st
import requests


# --------------------------------
# Configuration
# --------------------------------

BACKEND_URL = "http://127.0.0.1:8000"


# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Hybrid RAG",
    page_icon="📄",
    layout="centered"
)


# --------------------------------
# Page Title
# --------------------------------

st.title("📄 Hybrid RAG Assistant")

st.write(
    "Upload a PDF document. "
    "The document will be processed using Vector RAG and Knowledge Graph RAG."
)


# --------------------------------
# Initialize Session State
# --------------------------------

if "document_ready" not in st.session_state:
    st.session_state.document_ready = False

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None


# --------------------------------
# PDF Upload
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
    accept_multiple_files=False
)


# --------------------------------
# Send PDF to FastAPI
# --------------------------------

if uploaded_file is not None:

    st.info(f"Selected file: {uploaded_file.name}")

    if st.button("Upload & Process Document"):

        with st.spinner("Uploading document..."):

            try:

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf"
                    )
                }

                response = requests.post(
                    f"{BACKEND_URL}/documents/upload",
                    files=files,
                    timeout=300
                )

                if response.status_code == 200:

                    data = response.json()

                    st.success(
                        f"PDF uploaded successfully: {data['filename']}"
                    )

                    st.session_state.uploaded_filename = data["filename"]

                    # IMPORTANT:
                    st.session_state.document_ready = True

                else:

                    st.error(
                        f"Upload failed: {response.text}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Unable to connect to the FastAPI backend. "
                    "Make sure the backend is running."
                )

            except requests.exceptions.RequestException as error:

                st.error(
                    f"Backend request failed: {error}"
                )


# --------------------------------
# Document Status
# --------------------------------

st.divider()

st.subheader("Document Status")

if st.session_state.document_ready:

    st.success("Document is ready for chat.")

else:

    st.warning(
        "Upload and processing must complete before chatting."
    )


# --------------------------------
# Chat
# --------------------------------

st.divider()

st.subheader("💬 Chat with your document")

user_question = st.chat_input(
    "Ask a question about your PDF...",
    disabled=not st.session_state.document_ready
)

if user_question:

    # Display user's question
    st.chat_message("user").write(user_question)

    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={
                "question": user_question
            },
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            st.chat_message("assistant").write(
                data["answer"]
            )

        elif response.status_code == 400:
            data = response.json()
            st.warning(
                data.get(
                    "detail",
                    "Your request could not be processed."
                )
            )
        else:
            st.error(
                "Something went wrong while processing your question."
            )

    except requests.exceptions.RequestException as error:
        st.error(
            f"Backend request failed: {error}"
        )