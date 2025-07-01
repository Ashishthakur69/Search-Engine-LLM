import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import PromptTemplate


load_dotenv()

## Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search = TavilySearchResults()


st.title("🔎 LangChain - Chat with search")

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display message history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
# Chat input
if prompt:=st.chat_input(placeholder="Ask me anything..."):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
    tools=[search,arxiv,wiki]
    # Custom agent prompt to avoid parsing errors
    custom_prompt = PromptTemplate.from_template(
        """You are an AI assistant using tools to answer questions. Stick to this format exactly:

If using a tool:
Thought: <reasoning>
Action: <tool_name>
Action Input: <your input>

If answering finally:
Final Answer: <your answer>

ONLY use this format. Don't say anything else or explain the format.

Question: {input}
{agent_scratchpad}"""
    )
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True,agent_kwargs={"prompt": custom_prompt})

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)

