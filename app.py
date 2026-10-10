import streamlit as st
from router import app, config

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
        status = st.empty()
        placeholder = st.empty()
        status.markdown("_Thinking..._")

        full_response = ""
        
        for chunk in app.stream(
    {"messages": [{"role": "user", "content": query}]},
    config,
    stream_mode="custom",):
            if chunk["type"] == "routing":
                sid = chunk["specialist_id"]
                if sid != "general":
                    status.markdown(f"_Referring to the {sid.replace('-', ' ')}..._")
            elif chunk["type"] == "token":
                if not full_response:
                    status.empty()
                full_response += chunk["content"]
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.history.append(("assistant", full_response))
    st.rerun()