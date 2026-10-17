from pathlib import Path
import asyncio
import queue
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_rag_backend import (
    chatbot,
    retrieve_all_threads,
    submit_async_task,
    ingest_pdf,
    thread_document_metadata,
)


# =========================== Utilities ===========================

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    thread_id = str(thread_id)
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    add_thread(thread_id)


def load_conversation(thread_id):
    config = {"configurable": {"thread_id": str(thread_id)}}
    state = submit_async_task(chatbot.aget_state(config=config)).result()
    return state.values.get("messages", [])


def display_content(content):
    """Convert LangChain message content to something Streamlit can render."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content)


# ======================= Session Initialization ===================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = [
        str(thread_id) for thread_id in retrieve_all_threads()
    ]

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

st.session_state["thread_id"] = str(st.session_state["thread_id"])
add_thread(st.session_state["thread_id"])

thread_key = st.session_state["thread_id"]
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
selected_thread = None


# ============================ Sidebar ============================

st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename', 'PDF')}` "
        f"({latest_doc.get('chunks', '?')} chunks from "
        f"{latest_doc.get('documents', '?')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat",
    type=["pdf"],
    key=f"pdf-upload-{thread_key}",
)

if uploaded_pdf is not None:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            try:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )
                thread_docs[uploaded_pdf.name] = summary
                status_box.update(
                    label="✅ PDF indexed",
                    state="complete",
                    expanded=False,
                )
            except Exception as exc:
                status_box.update(
                    label="PDF indexing failed",
                    state="error",
                    expanded=True,
                )
                st.sidebar.error(str(exc))

st.sidebar.subheader("Past conversations")
threads = list(reversed(st.session_state["chat_threads"]))

if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for past_thread_id in threads:
        if st.sidebar.button(
            str(past_thread_id),
            key=f"side-thread-{past_thread_id}",
            use_container_width=True,
        ):
            selected_thread = str(past_thread_id)


# ============================ Main Layout ========================

st.title("Multi Utility Chatbot")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    event_queue = queue.Queue()
    assistant_parts = []
    tool_names = []

    async def produce_stream():
        try:
            async for chunk, metadata in chatbot.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                event_queue.put(("chunk", chunk))
        except Exception as exc:
            event_queue.put(("error", exc))
        finally:
            event_queue.put(("done", None))

    future = submit_async_task(produce_stream())

    with st.chat_message("assistant"):
        status_box = None

        def ai_only_stream():
            nonlocal_status = {"box": None}
            while True:
                event_type, payload = event_queue.get()

                if event_type == "done":
                    break

                if event_type == "error":
                    raise payload

                message_chunk = payload

                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", None) or "tool"
                    if nonlocal_status["box"] is None:
                        nonlocal_status["box"] = st.status(
                            f"🔧 Using `{tool_name}`…",
                            expanded=True,
                        )
                    else:
                        nonlocal_status["box"].update(
                            label=f"🔧 Using `{tool_name}`…",
                            state="running",
                            expanded=True,
                        )
                    continue

                if isinstance(message_chunk, AIMessage):
                    content = display_content(message_chunk.content)
                    if content:
                        assistant_parts.append(content)
                        yield content

            if nonlocal_status["box"] is not None:
                nonlocal_status["box"].update(
                    label="✅ Tool activity finished",
                    state="complete",
                    expanded=False,
                )

        try:
            st.write_stream(ai_only_stream())
            future.result()
        except Exception as exc:
            st.error(f"Chatbot error: {exc}")

    assistant_response = "".join(assistant_parts)
    if assistant_response:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": assistant_response}
        )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename', 'PDF')} "
            f"(chunks: {doc_meta.get('chunks', '?')}, "
            f"pages: {doc_meta.get('documents', '?')})"
        )

st.divider()

# Switch conversations after rendering the current one.
if selected_thread and selected_thread != thread_key:
    try:
        restored_messages = load_conversation(selected_thread)
        restored_history = []

        for msg in restored_messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                # Avoid displaying tool-call-only assistant messages as blank bubbles.
                if not display_content(msg.content).strip():
                    continue
                role = "assistant"
            else:
                continue

            restored_history.append(
                {"role": role, "content": display_content(msg.content)}
            )

        st.session_state["thread_id"] = selected_thread
        st.session_state["message_history"] = restored_history
        st.session_state["ingested_docs"].setdefault(selected_thread, {})
        st.rerun()
    except Exception as exc:
        st.error(f"Could not load conversation: {exc}")

