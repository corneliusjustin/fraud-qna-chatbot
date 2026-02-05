import streamlit as st


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False


def render_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "metadata" in msg:
                _render_metadata(msg["metadata"])


def _render_metadata(metadata: dict):
    cols = st.columns([1, 1, 2])
    with cols[0]:
        score = metadata.get("score", 0)
        color = "green" if score >= 4 else "orange" if score >= 3 else "red"
        st.markdown(f"**Quality:** :{color}[{score}/5]")
    with cols[1]:
        qtype = metadata.get("query_type", "unknown")
        badge = {"sql": "📊", "rag": "📄", "hybrid": "📊📄"}.get(qtype, "❓")
        st.markdown(f"**Type:** {badge} {qtype.upper()}")


def add_user_message(content: str):
    st.session_state.messages.append({"role": "user", "content": content})


def add_assistant_message(content: str, metadata: dict | None = None):
    msg = {"role": "assistant", "content": content}
    if metadata:
        msg["metadata"] = metadata
    st.session_state.messages.append(msg)
