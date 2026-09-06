import streamlit as st
from query import answer_question


st.title("Audio Deepfake Research Assistant")
st.caption("RAG system over a corpus of audio deepfake detection papers")

question = st.text_input("Ask a question about audio deepfake detection:")


if question:
    with st.spinner("Retrieving and generation answer . . ."):
        answer, citations = answer_question(question)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for i, c in enumerate(citations):
        with st.expander(f"[{i+1}] {c['source']} (distance: {c['distance']})"):
            st.write(c["excerpt"] + "...")
