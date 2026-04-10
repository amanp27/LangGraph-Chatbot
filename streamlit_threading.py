import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

from dotenv import load_dotenv
load_dotenv()

# UTILS ----------------------------------------------------------------

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id


# SESSION STATE ---------------------------------------------------------

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()


# SIDEBAR ---------------------------------------------------------------

st.sidebar.title("LangGraph Chatbot")
st.sidebar.button("New Chat")
st.sidebar.header("Chat History")

st.sidebar.text(st.session_state['thread_id'])


CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}


for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("Type here...")

if user_input:

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    with st.chat_message("assistant"):

        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': HumanMessage(content = user_input)},
                config = CONFIG,
                stream_mode = 'messages'
                )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})