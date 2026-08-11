import streamlit as st
from router import app  # your compiled LangGraph app

st.set_page_config(page_title="Model Router", page_icon="🧭")
st.title("Specialist Model Router")

if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

query = st.chat_input("Ask something...")

if query:
    st.session_state.history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for chunk in app.stream({"query": query}, stream_mode="custom"):
            full_response += chunk
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    st.session_state.history.append(("assistant", full_response))