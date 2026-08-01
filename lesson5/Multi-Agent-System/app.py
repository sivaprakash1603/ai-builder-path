import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()

# We import the graph after loading dotenv so it picks up the API key
from graph import graph

st.set_page_config(page_title="Multi-Agent Support", page_icon="🤖", layout="centered")

st.title("🏢 Enterprise Support Desk")
st.markdown("Ask our intelligent agents your **IT** or **Finance** questions. Our Supervisor will seamlessly route your request to the right department.")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        agent_name = getattr(message, 'name', None) or 'Assistant'
        avatar = "💻" if agent_name == "IT" else "💰" if agent_name == "Finance" else "🤖"
        with st.chat_message("assistant", avatar=avatar):
            st.markdown(f"**{agent_name} Agent**: {message.content}")

# Handle new user input
if prompt := st.chat_input("e.g., How do I set up a VPN?"):
    
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Spinner while graph executes
    with st.spinner("Our supervisor is routing your request..."):
        initial_state = {"messages": st.session_state.messages}
        
        try:
            # Stream the LangGraph execution
            for event in graph.stream(initial_state, {"recursion_limit": 10}):
                for node_name, state_update in event.items():
                    # If it's a worker node returning a message, display it!
                    if node_name in ["IT", "Finance"] and "messages" in state_update:
                        for message in state_update["messages"]:
                            
                            # Append to session state
                            st.session_state.messages.append(message)
                            
                            agent_name = getattr(message, 'name', None) or node_name
                            avatar = "💻" if agent_name == "IT" else "💰" if agent_name == "Finance" else "🤖"
                            
                            with st.chat_message("assistant", avatar=avatar):
                                st.markdown(f"**{agent_name} Agent**: {message.content}")
                                
        except Exception as e:
            st.error(f"An error occurred: {e}")
