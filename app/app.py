"""
Write a streamlit application that takes a user input and returns response as per the user query written in app.py
"""

import json
from pprint import pprint
from time import time
import os
from langchain_core.messages import (
    HumanMessage,
)
from swarm.graph import initialize_swarm_graph
from swarm.utils import get_llm
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from dotenv import load_dotenv
load_dotenv()

os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "true"
CUSTOM_HTML = """
<div class="banner">
    <img src="./app/static/MDB-langchain.jpg" alt="MongoDB + Langchain Img">
</div>
<style>
    .banner {
        width: auto;
        height: auto;
        overflow-wrap: break-word;
    }
    .banner img {
        width: 512px;
        height: auto;
        align: center;
        object-fit: cover;
    }
</style>
"""


def get_chat_response(question, graph, user_session: str) -> str:
    config = {"configurable": {"thread_id": user_session}, "recursion_limit": 15}
    resp = graph.stream(
        {
            "messages": [
                HumanMessage(
                    content=question
                )
            ],
        },
        config=config
    )

    # Display the response in the chat message container
    start_time = time()
    response = []
    for output in resp:
        for key,value in output.items():
            st.subheader("Response")
            expander = st.expander("Expand to view agent reflections")
            expander.markdown(f"{key} : {value['messages'][-1].content}")
            expander.write(
                f"Time taken to respond: {time() - start_time} seconds")
            pprint(f"Output from node '{key}':")
            pprint("---")
            pprint(value, indent=2, width=80, depth=None)
            response += [(key,value)]
    return response[-1][1]["messages"][-1].content,response

# Create two tabs
tab, tab1, tab2 = st.tabs(["Banner", "Architecture", "Application"])
with tab:
    st.markdown(CUSTOM_HTML, unsafe_allow_html=True)
with tab1:
    st.markdown(
        """<img src="./app/static/agents_nodes_graph.png.jpg" class=" css-1lo3ubz" alt="Architecture logo" style="height:auto;width:auto;align:center"> """,
        unsafe_allow_html=True)

with tab2:
    with st.sidebar:
        st.title("Multi Agent RAG System", "🔍")
        # table of images side by side
        st.markdown(
            """<img src="./app/static/legal_advisor_iteration1.jpg" class=" css-1lo3ubz" alt="Architecture logo" style="height:auto;width:auto"> """,
            unsafe_allow_html=True)

        # Dropdown to choose an LLM model with a button to refresh the model
        add_vertical_space(1)
        llm_model_expander = st.expander("Choose yor LLM model")
        CHOICES = {"accounts/fireworks/models/llama-v3p1-405b-instruct": "llama3.1-405B",
                   "accounts/fireworks/models/llama-v3p1-70b-instruct": "llama-3.1-70B",
                   "accounts/fireworks/models/mixtral-8x7b-instruct": "mixtral-8x7B",
                   "accounts/fireworks/models/qwen2p5-72b-instruct": "qwen2.5-72B"}

        def format_func(option):
            return CHOICES[option]
        model = llm_model_expander.selectbox(
            "Select option", options=list(CHOICES.keys()), format_func=format_func)
        add_vertical_space(1)
        st.write(f"You selected option {model} called {format_func(model)}")
        add_vertical_space(1)
        add_vertical_space(3)
        st.session_state.user_session = st.text_input("Enter your user name", value="John Doe")
        add_vertical_space(1)
        add_vertical_space(3)
        if llm_model_expander.button("Refresh Model"):
            st.session_state.model = model
            graph = initialize_swarm_graph(st.session_state.model)
            st.session_state.graph = graph
        
        # Initialize user session name


    if model not in st.session_state or st.session_state.model is None:
        st.session_state.model = get_llm()
        st.session_state.graph = initialize_swarm_graph(st.session_state.model)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me Anything!!!"):

        if st.session_state.graph is None:
            st.session_state.graph = initialize_swarm_graph(
                st.session_state.model)
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response,full_response = get_chat_response(prompt, st.session_state.graph, st.session_state.user_session)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response})
