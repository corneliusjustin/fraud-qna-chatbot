import streamlit as st
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from services.database import is_database_ready
from services.vector_store import is_vector_store_ready
from components.chat_interface import (
    init_session_state,
    render_chat_history,
    add_user_message,
    add_assistant_message,
)
from components.response_display import render_response
from components.quality_indicator import render_quality_badge
from core.agent import process_query_stream, AgentStep
from models.schemas import AgentResponse

# Page config
st.set_page_config(
    page_title="Fraud Analysis Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage { max-width: 100%; }
    .block-container { padding-top: 2rem; }
    div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

init_session_state()


# Sidebar
with st.sidebar:
    st.title("🔍 Fraud Analysis Agent")
    st.markdown("---")

    # System status
    st.subheader("System Status")
    db_ready = is_database_ready()
    vs_ready = is_vector_store_ready()

    st.markdown(f"{'✅' if db_ready else '❌'} **SQLite Database**")
    st.markdown(f"{'✅' if vs_ready else '❌'} **ChromaDB Vector Store**")

    if not db_ready or not vs_ready:
        st.warning("Data sources not initialized. Run `python scripts/setup_data.py` first.")

    st.markdown("---")

    # Sample questions
    st.subheader("💡 Sample Questions")
    sample_questions = [
        "How does the monthly fraud rate fluctuate over the two-year period?",
        "Which merchant categories exhibit the highest incidence of fraudulent transactions?",
        "What are the primary methods by which credit card fraud is committed?",
        "What are the core components of an effective fraud detection system?",
        "How much higher are fraud rates when the counterpart is outside the EEA?",
        "What share of total card fraud value in H1 2023 was due to cross-border transactions?",
    ]

    for i, q in enumerate(sample_questions):
        if st.button(f"Q{i+1}", key=f"sample_{i}", help=q, use_container_width=True):
            st.session_state.pending_question = q

    st.markdown("---")
    st.caption("Built with Together AI, SQLite & ChromaDB")


# Main chat area
st.header("💬 Chat")

render_chat_history()

# Handle pending question from sidebar
pending = st.session_state.pop("pending_question", None)

# Chat input
user_input = st.chat_input("Ask a question about fraud data or the fraud report...")

question = pending or user_input

if question:
    add_user_message(question)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Status container for step-by-step updates
        status_container = st.status("Processing your question...", expanded=True)
        # Placeholder for streamed answer text
        answer_placeholder = st.empty()
        # Will hold the final AgentResponse
        final_response = None
        streamed_text = ""
        is_streaming = False

        # Build conversation history for context
        chat_history = []
        for msg in st.session_state.messages:
            chat_history.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        for event in process_query_stream(question, history=chat_history):
            if isinstance(event, AgentStep):
                # Show pipeline step in the status expander
                status_container.update(label=event.label, state="running")
                if event.detail:
                    status_container.write(f"  _{event.detail}_")
                else:
                    status_container.write(event.label)

                # When we start synthesizing, switch to streaming mode
                if event.step == "synthesize":
                    is_streaming = True

                # When scoring starts, stop streaming mode
                if event.step == "score":
                    is_streaming = False

            elif isinstance(event, str):
                # Streaming token from LLM
                streamed_text += event
                answer_placeholder.markdown(streamed_text + "▌")

            elif isinstance(event, AgentResponse):
                final_response = event

        # Finalize the streamed text (remove cursor)
        if streamed_text:
            answer_placeholder.markdown(streamed_text)

        if final_response:
            if final_response.error and not final_response.answer:
                status_container.update(label="❌ Error", state="error", expanded=False)
                st.error(f"Error: {final_response.error}")
                add_assistant_message(f"Error: {final_response.error}")
            else:
                # Collapse the status and mark complete
                score = final_response.quality_score.score if final_response.quality_score else 0
                score_emoji = "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
                status_container.update(
                    label=f"Done — {score_emoji} Quality: {score}/5",
                    state="complete",
                    expanded=False,
                )

                # Quality badge
                render_quality_badge(score, final_response.query_type.value)

                # Render charts, sources, quality details (but NOT the answer text again)
                render_response(final_response, skip_answer=True)

                # Store in chat history
                metadata = {
                    "score": score,
                    "query_type": final_response.query_type.value,
                }
                add_assistant_message(final_response.answer, metadata=metadata)
