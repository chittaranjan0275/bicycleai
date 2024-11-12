import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
import os
import asyncio

async def chat_with_data(db_path: str, query: str):
    """Chat with the data using LangChain SQL Agent."""
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    llm = ChatOpenAI(temperature=0)
    
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True
    )
    
    result = await agent_executor.ainvoke({"input": query})
    return result['output']

def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'db_path' not in st.session_state:
        db_name = "aviation_test.db"
        st.session_state.db_path = os.path.join(os.getcwd(), db_name)

def main():
    st.set_page_config(page_title="Aviation Database Chat", layout="wide")
    st.title("Aviation Database Chat Interface")
    
    init_session_state()
    
    # Check if database exists
    if os.path.exists(st.session_state.db_path):
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your data"):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = asyncio.run(chat_with_data(st.session_state.db_path, prompt))
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    else:
        st.error("Database not found. Please ensure aviation_test.db exists in the current directory.")

if __name__ == "__main__":
    main() 