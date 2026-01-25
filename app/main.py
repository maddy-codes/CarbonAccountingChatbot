import os
import sys

import streamlit as st

# fmt: off
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.generator import CarbonChatbot

# fmt: on

# Page Config
st.set_page_config(page_title="Carbon Accounting AI", page_icon="🌱")


# Initialize the chatbot (cached so it only loads once)
@st.cache_resource
def load_bot():
    return CarbonChatbot()


bot = load_bot()

# UI Layout [cite: 228]
st.title("🌱 Intelligent Carbon Accounting Chatbot")
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history [cite: 226]
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message:
            st.caption(f"Sources: {', '.join(message['citations'])}")

# Chat Input [cite: 223]
if prompt := st.chat_input("Ask a carbon accounting question..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching regulations..."):
            response = bot.ask(prompt)
            full_answer = response["answer"]
            citations = response["citations"]

            st.markdown(full_answer)
            st.caption(f"Sources: {', '.join(citations)}")

            # Save assistant response
            st.session_state.messages.append(
                {"role": "assistant", "content": full_answer, "citations": citations}
            )
