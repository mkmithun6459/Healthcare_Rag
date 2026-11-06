
import streamlit as st
import sys
import os

# Make src modules accessible
sys.path.append(
    os.path.join(os.path.dirname(__file__), "src")
)

from rag_pipeline import generate_answer


# Page configuration
st.set_page_config(
    page_title="MediAssist - Healthcare AI",
    page_icon="🩺",
    layout="centered"
)


# Custom styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #16A085;
    }

    .subtitle {
        text-align: center;
        color: gray;
        font-size: 16px;
    }

    .disclaimer {
        background-color: #FFF3CD;
        padding: 12px;
        border-radius: 8px;
        color: #664D03;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# Header
st.markdown(
    '<div class="main-title">🩺 MediAssist</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered Medical Information Assistant</div>',
    unsafe_allow_html=True
)

st.write("")

st.markdown("""
<div class="disclaimer">
⚠️ MediAssist provides general medical information, not
medical diagnosis or treatment. Consult a qualified
healthcare professional for personal medical concerns.
</div>
""", unsafe_allow_html=True)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View medical sources"):
                for source in message["sources"]:
                    st.write("**Source:**", source.get("source", "Unknown"))
                    st.write("**Topic:**", source.get("topic", "Unknown"))
                    st.write("**Distance:**", round(source.get("distance", 0), 4))
                    st.write("**URL:**", source.get("source_url", "Unavailable"))


# Chat input
question = st.chat_input("Ask your medical question...")


if question:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Searching medical knowledge..."):

            try:
                answer, results = generate_answer(question)

                st.markdown(answer)

                sources = [
                    result["metadata"] | {
                        "distance": result["distance"]
                    }
                    for result in results
                ]

                if sources:
                    with st.expander("View medical sources"):
                        for source in sources:
                            st.write("**Source:**", source.get("source", "Unknown"))
                            st.write("**Topic:**", source.get("topic", "Unknown"))
                            st.write("**Distance:**", round(source["distance"], 4))
                            st.write("**URL:**", source.get("source_url", "Unavailable"))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")


# Sidebar
with st.sidebar:

    st.title("MediAssist")

    st.write("Medical knowledge assistant powered by:")

    st.write("- MedQuAD dataset")
    st.write("- SentenceTransformers")
    st.write("- ChromaDB")
    st.write("- Ollama Llama 3")

    st.divider()

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
