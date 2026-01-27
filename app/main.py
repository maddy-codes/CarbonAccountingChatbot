"""
Main application entry point for the Carbon Accounting Chatbot.

This Streamlit app provides the user interface for interacting with the chatbot.
"""
import os
import sys

# Configure environment before importing heavy libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import torch

# Add project root to path to allow imports from src
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.generator import AzureCarbonChatbot

# Page Configuration
st.set_page_config(page_title="Carbon Accounting AI")


@st.cache_resource
def load_bot():
    """Initializes and caches the chatbot instance."""
    return AzureCarbonChatbot()


bot = load_bot()

# UI Layout
st.title("Intelligent Carbon Accounting Chatbot")
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message:
            st.caption(f"Sources: {', '.join(message['citations'])}")

# Chat Input
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
