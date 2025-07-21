import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory

## Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=1000)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="Search")


st.title("🔎🤖AI Chat with search🤖")

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")
if st.sidebar.button("Clear Chat History"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]
    st.rerun() # Rerun the app to clear chat display

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":"Hi,🤖 I'm a chatbot who can search the web and arxiv . How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Initialize memory for the chatbot
# k=5 means it will remember the last 5 turns of conversation
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)


if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
    tools=[search,arxiv,wiki]

    search_agent=initialize_agent(tools,llm,agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,memory=memory,handle_parsing_errors=True)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."): # Added loading spinner
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(prompt,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)

