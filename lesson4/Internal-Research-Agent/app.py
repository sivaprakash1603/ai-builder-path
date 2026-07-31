import streamlit as st
from agent import create_agent, SYSTEM_PROMPT

st.set_page_config(page_title="Presidio Research Agent", page_icon="🕵️‍♂️", layout="centered")

st.title("🕵️‍♂️ Presidio Internal Research Agent")
st.markdown("Ask me anything about internal HR policies, industry benchmarks, or Presidio Google Docs!")

# Initialize agent only once
if "agent" not in st.session_state:
    with st.spinner("Initializing Agent..."):
        try:
            st.session_state.agent = create_agent()
            st.success("Agent is ready!")
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question (e.g., 'Compare hiring trends' or 'Find HR compliance policies')"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # We need to construct the message history for LangGraph
                # We always start with the system prompt
                history = [("system", SYSTEM_PROMPT)]
                
                # Add all previous chat history so it remembers context
                for msg in st.session_state.messages:
                    history.append((msg["role"], msg["content"]))
                
                # Invoke the agent
                response = st.session_state.agent.invoke({"messages": history})
                final_message = response["messages"][-1].content
                
                st.markdown(final_message)
                st.session_state.messages.append({"role": "assistant", "content": final_message})
            except Exception as e:
                st.error(f"Error: {str(e)}")
